import argparse
import gc
import json
import os
import random
import sys

import numpy as np
import torch
from pyod.models.vae import VAE
from sklearn.cluster import KMeans
from tqdm import tqdm

from utils.dataset import dataset
from models.model import *
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DfauLo():
    def __init__(self, args):
        self.args = args

    def run(self, data_s):
        model = torch.load(self.args.model)
        vae_del_list, km_del_list, loss_del_list = self.OAL(data_s)

        model_vae = self.mutation(vae_del_list, data_s)
        model_km = self.mutation(km_del_list, data_s)
        model_loss = self.mutation(loss_del_list, data_s)

        image_list, gt_list, org_SFM_list, org_Loss_list = zip(*self.get_feature(model, data_s))

        with open(os.path.join(self.args.dataset, 'classes.json'), 'r') as f:
            classes = json.load(f)
        class_num = len(classes.keys())

        # sample 10 index for each gt(class_num gts)
        random_index = []
        for i in range(class_num):
            index = np.where(np.array(gt_list) == i)[0]
            random_index.extend(random.sample(index.tolist(), 10))

        # transform gt_list to one-hot
        gt_one_hot = np.zeros((len(gt_list), class_num))
        for i, gt in enumerate(gt_list):
            gt_one_hot[i][gt] = 1
        gt_list = gt_one_hot.tolist()

        _, _, vae_SFM_list, vae_Loss_list = zip(*self.get_feature(model_vae, data_s))
        _, _, km_SFM_list, km_Loss_list = zip(*self.get_feature(model_km, data_s))
        _, _, loss_SFM_list, loss_Loss_list = zip(*self.get_feature(model_loss, data_s))

        Feature = [[*org_SFM, *gt, *vae_SFM, *km_SFM, *loss_SFM, (1 if img in vae_del_list else 0),
                    (1 if img in km_del_list else 0), (1 if img in loss_del_list else 0), org_Loss, vae_Loss, km_Loss,
                    loss_Loss]
                   for img, gt, org_SFM, vae_SFM, km_SFM, loss_SFM, org_Loss, vae_Loss, km_Loss, loss_Loss in
                   zip(image_list, gt_list, org_SFM_list, vae_SFM_list, km_SFM_list, loss_SFM_list, org_Loss_list,
                       vae_Loss_list, km_Loss_list, loss_Loss_list)]
        Feature = np.array(Feature)
        print('Feature shape: ', Feature.shape)

        sample_feature = Feature[random_index]
        print('sample_feature shape: ', sample_feature.shape)

        # random shuffle Feature and corresponding image_list
        image_list = np.array(image_list)
        idx_shuffle = [i for i in range(len(Feature))]
        random.seed(2023)
        random.shuffle(idx_shuffle)
        Feature = Feature[idx_shuffle]
        image_list = image_list[idx_shuffle]

        random_Feature = self.getrandomfeature(model, model_vae, model_km, model_loss, class_num)
        random_Feature = np.array(random_Feature)

        print('random Feature shape:', random_Feature.shape)

        sample_feature = np.concatenate((sample_feature, random_Feature), axis=0)
        print('sample_feature merged shape: ', Feature.shape)

        Y = [0 for i in range(10 * class_num)]  # ground truth: 10*class_num = 0 , class_num = 1
        Y.extend([1 for i in range(class_num)])

        Y = np.array(Y)
        print('Y shape: ', Y.shape)

        lg = LogisticRegression(C=1.0)
        lg.fit(sample_feature, Y)

        LRres = lg.predict_proba(Feature)
        LRres = LRres[:, 1]

        print('LRres shape: ', LRres.shape)

        # sort image_list by LRres in descending order

        image_list = image_list[np.argsort(-LRres)]

        image_list = image_list[:int(self.args.top_ratio * len(image_list))]

        print('output located image_list shape: ', image_list.shape)

        # save memory
        del model, model_vae, model_km, model_loss, vae_del_list, km_del_list, \
            loss_del_list, gt_list, org_SFM_list, org_Loss_list, vae_SFM_list, vae_Loss_list, km_SFM_list, \
            km_Loss_list, loss_SFM_list, loss_Loss_list, Feature, sample_feature, random_Feature, Y, lg, LRres

        return image_list.tolist()

    def OAL(self, data_s):
        args = self.args
        model = torch.load(args.model)
        modelargs = torch.load(args.model_args)
        loss_fn = modelargs['loss_fn']

        with open(os.path.join(args.dataset, 'classes.json'), 'r') as f:
            classes = json.load(f)
        class_keys = list(classes.keys())
        vae_del_list, km_del_list, loss_del_list = [], [], []
        for specific_label in class_keys:
            print('\nrunning Outlier-VAE on label: ', specific_label)
            dataset_ = dataset(root=args.dataset, classes_path='classes.json', transform=modelargs['transform'],
                               image_size=eval(args.image_size), image_set=args.image_set,
                               specific_label=specific_label, data_s=data_s)
            data_loader = torch.utils.data.DataLoader(dataset_, batch_size=1, shuffle=False, num_workers=0)

            def dataset_generator():
                for sample in dataset_:
                    yield sample[0].numpy().reshape(1, -1)

            # train VAE
            vae = VAE(epochs=1, verbose=False)
            vae.n_features_ = eval(args.image_size)[0] * eval(args.image_size)[1] * eval(args.image_size)[2]
            vae = vae._build_model()

            train_tfdataset = tf.data.Dataset.from_generator(generator=dataset_generator,
                                                             output_types=tf.float32,
                                                             output_shapes=tf.TensorShape([None, None]))
            test_tfdataset = tf.data.Dataset.from_generator(generator=dataset_generator,
                                                            output_types=tf.float32,
                                                            output_shapes=tf.TensorShape([None, None]))

            vae.fit(train_tfdataset, epochs=1)
            pred_scores = vae.predict(test_tfdataset)
            # covert test_tfdataset to numpy array

            vae_decision_scores_ = self.pairwise_distances_no_broadcast(test_tfdataset,
                                                                        pred_scores)

            print('\nrunning Activation-Clustering and Loss-sorting on label: ', specific_label)

            def get_features_hook(module, input, output):
                global features
                features = output

            moudle = getattr(model, args.hook_layer)
            hook_handle = moudle.register_forward_hook(get_features_hook)
            model.to(device)
            act_features, image_list, Loss_list = [], [], []
            softmax_func = nn.Softmax(dim=1)
            model.eval()
            with torch.no_grad():
                for i, (images, labels, image_paths) in enumerate(data_loader):
                    out = model(images.to(device))
                    labels = labels.to(device)
                    print('\r', 'processing image: ', i, end='')
                    act_features.append(features.cpu().view(-1).numpy().tolist())
                    Loss = loss_fn(softmax_func(out), labels).cpu().numpy().item()

                    Loss_list.append(Loss)
                    image_list.append(image_paths[0])

            clf_kmeans = KMeans(n_clusters=2)
            clf_kmeans.fit(act_features)
            km_label = clf_kmeans.labels_

            # sort image_list according to vae_decision_scores_ in descending order
            zip_list = list(zip(image_list, vae_decision_scores_, km_label, Loss_list))
            zip_list.sort(key=lambda x: x[1], reverse=True)
            image_list, vae_decision_scores_, km_label, Loss_list = zip(*zip_list)

            tmp_vae_del_list = image_list[:int(len(image_list) * float(args.rm_ratio))]
            if sum(km_label) > len(km_label) / 2:
                tmp_km_del_list = [image_list[i] for i in range(len(image_list)) if km_label[i] == 0]
            else:
                tmp_km_del_list = [image_list[i] for i in range(len(image_list)) if km_label[i] == 1]
            zip_list.sort(key=lambda x: x[3], reverse=True)
            image_list, vae_decision_scores_, km_label, Loss_list = zip(*zip_list)
            tmp_loss_del_list = image_list[:int(len(image_list) * float(args.rm_ratio))]
            vae_del_list.extend(tmp_vae_del_list)
            km_del_list.extend(tmp_km_del_list)
            loss_del_list.extend(tmp_loss_del_list)
        return vae_del_list, km_del_list, loss_del_list

    def mutation(self, del_list, data_s):
        args = self.args
        print('mutation start')
        model = torch.load(args.model)
        modelargs = torch.load(args.model_args)
        loss_fn = modelargs['loss_fn']
        optimizer = modelargs['optimizer']
        datasets = dataset(root=args.dataset, classes_path='classes.json', transform=modelargs['transform'],
                           image_size=eval(args.image_size), image_set=args.image_set,
                           ignore_list=del_list, data_s=data_s)
        data_loader = torch.utils.data.DataLoader(datasets, batch_size=args.retrain_bs, shuffle=True, num_workers=0)

        model.to(device)
        model.train()
        for epoch in range(args.retrain_epoch):
            for i, (images, labels, image_paths) in enumerate(data_loader):
                out = model(images.to(device))
                labels = labels.to(device)
                loss = loss_fn(out, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('\r', 'epoch: ', epoch, 'processing batch: ', i, end='')
        return model

    def get_feature(self, model, data_s):
        args = self.args
        modelargs = torch.load(args.model_args)
        datasets = dataset(root=args.dataset, classes_path='classes.json', transform=modelargs['transform'],
                           image_size=eval(args.image_size), image_set=args.image_set, data_s=data_s)
        data_loader = torch.utils.data.DataLoader(datasets, batch_size=1, shuffle=False, num_workers=0)
        loss_fn = modelargs['loss_fn']
        softmax_func = nn.Softmax(dim=1)
        model.eval()
        model.to(device)
        SFM_list, Loss_list, image_list, gt_list = [], [], [], []
        with torch.no_grad():
            for i, (images, labels, image_paths) in enumerate(data_loader):
                out = model(images.to(device))
                labels = labels.to(device)
                Loss = loss_fn(softmax_func(out), labels).cpu().numpy()
                SFM = softmax_func(out).cpu().numpy()[0]
                Loss_list.append(Loss.item())
                SFM_list.append(SFM.tolist())
                gt_list.append(labels.cpu().numpy()[0])
                image_list.append(image_paths[0])

        return zip(image_list, gt_list, SFM_list, Loss_list)

    def getrandomfeature(self, model_org, model_vae, model_km, model_loss, class_num):
        modelargs = torch.load(args.model_args)

        def model_out(model, X, Y):
            model.eval()
            loss_fn = modelargs['loss_fn']
            softmax_func = nn.Softmax(dim=1)
            with torch.no_grad():
                X = X.to(device)
                out = model(X)
                y = torch.from_numpy(np.array([Y])).long().to(device)
                soft_output = softmax_func(out)
                loss = loss_fn(soft_output, y).cpu().numpy().item()
                sfout = soft_output.cpu().numpy()[0].tolist()
            return sfout, loss

        image_size = eval(args.image_size)
        X = torch.rand(1, image_size[2], image_size[0], image_size[1])
        Feature = []
        for i in range(class_num):
            sfout_org, loss_org = model_out(model_org, X, i)
            sfout_vae, loss_vae = model_out(model_vae, X, i)
            sfout_km, loss_km = model_out(model_km, X, i)
            sfout_loss, loss_loss = model_out(model_loss, X, i)
            gt = np.zeros(class_num)
            gt[i] = 1
            gt = gt.tolist()

            tmp_feature = [*sfout_org, *gt, *sfout_vae, *sfout_km, *sfout_loss, 1, 1, 1, loss_org, loss_vae, loss_km,
                           loss_loss]
            Feature.append(tmp_feature)
        return Feature

    def pairwise_distances_no_broadcast(self, X, Y):  # 可以进一步优化
        # X is a tf.data.Dataset
        X_array = []
        for x in X:
            X_array.append(x.numpy().reshape(-1))
        X_array = np.array(X_array)
        euclidean_sq = np.square(Y - X_array)
        del X_array
        return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()


def data_slice(args):
    slice_num = args.slice_num
    random.seed(2023)
    with open(os.path.join(args.dataset, 'classes.json'), 'r') as f:
        classes = json.load(f)
    class_keys = list(classes.keys())
    result = {i: {} for i in range(slice_num)}
    for name in class_keys:
        img_list_dir = os.listdir(os.path.join(args.dataset + '/' + args.image_set, name))
        img_list = []
        for img in img_list_dir:
            path = os.path.join(name, img)
            img_list.append(path)
        random.shuffle(img_list)
        slice_len = len(img_list) // slice_num
        for i in range(slice_num):
            result[i][name] = img_list[i * slice_len:(i + 1) * slice_len]
    print('data slice done with slice num: ', slice_num)
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='./dataset/mnist', help='input dataset')
    parser.add_argument('--model', default='./models/model.pth', help='input model path')
    parser.add_argument('--image_size', default='(28,28,1)', help='input image size')
    parser.add_argument('--model_args', default='./models/model_args.pth', help='input model args path')
    parser.add_argument('--image_set', default='train', help='input image set')
    parser.add_argument('--hook_layer', default='s4', help='input hook layer')
    parser.add_argument('--rm_ratio', default=0.05, help='input ratio')
    parser.add_argument('--top_ratio', default=0.01, help='input ratio')
    parser.add_argument('--retrain_epoch', default=1, help='input retrain epoch')
    parser.add_argument('--retrain_bs', default=64, help='input retrain batch size')
    parser.add_argument('--slice_num', default=1, help='input slice num')

    args = parser.parse_args()

    data_s = data_slice(args)
    results = []
    df = DfauLo(args)
    for i in range(args.slice_num):
        print('\n====================slice {} / {}====================\n'.format(i + 1, args.slice_num))
        slice_res=df.run(data_s[i])
        results.extend(slice_res)
        # clear gpu cache
        torch.cuda.empty_cache()
        # clear memory
        gc.collect()


    with open(os.path.join(args.dataset, 'located_image_list.txt'), 'w') as f:
        for img in results:
            f.write(img + '\r')
