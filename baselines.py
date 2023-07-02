import json
import os
import random
import time

import cleanlab
import torch
from torch import nn

from utils.dataset import *
from utils.simifeat import *
from utils.NCNV import *
from utils.models import *
import numpy as np
from deepod.models.dif import DeepIsolationForest


class Baselines():
    def __init__(self, args):
        self.args = args

    def MSP(self):
        if os.path.exists(
                os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/MSP_results_list.json')):
            MSP_results_list = self.load_json(
                os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/MSP_results_list.json'))
        else:
            gt_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/gt_list.json'))
            org_SFM_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/org_SFM_list.json'))
            image_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/image_list.json'))

            MSP_scores = []
            for i in range(len(gt_list)):
                predicted_label = np.argmax(org_SFM_list[i])
                MSP_scores.append(org_SFM_list[i][predicted_label])
            # sort image_list by MSP_scores in Increasing order
            idx = np.argsort(MSP_scores)
            MSP_results_list = np.array(image_list)[idx].tolist()
            for i in range(len(MSP_results_list)):
                MSP_results_list[i] = MSP_results_list[i].split('\\')[-1]
            self.save_as_json(MSP_results_list,
                              os.path.join(self.args.dataset,
                                           'results/' + self.args.model_name + '/MSP_results_list.json'))
        return MSP_results_list

    def DIF(self, data_s):
        DIF_time = -1
        if os.path.exists(
                os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/DIF_results_list.json')) and \
                os.path.exists(
                    os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/DIF_sorted_score_list.json')):
            DIF_results_list = self.load_json(
                os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/DIF_results_list.json'))
            DIF_sorted_score_list = self.load_json(
                os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/DIF_sorted_score_list.json'))

        else:
            start_time = time.time()
            with open(self.args.class_path, 'r') as f:
                classes = json.load(f)
            class_keys = list(classes.keys())
            DIF_scores, image_list = [], []
            for specific_label in class_keys:
                modelargs = torch.load(self.args.model_args)
                dataset_ = dataset(root=self.args.dataset, classes_path=self.args.class_path,
                                   transform=modelargs['transform'],
                                   image_size=eval(self.args.image_size), image_set=self.args.image_set,
                                   specific_label=specific_label, data_s=data_s)
                data_loader = torch.utils.data.DataLoader(dataset_, batch_size=1, shuffle=False, num_workers=0)
                dif_data = []

                for i, (images, labels, image_paths) in enumerate(data_loader):
                    dif_data.append(images.cpu().numpy().reshape(-1).tolist())
                    image_list.append(image_paths[0])
                dif_data = np.array(dif_data)
                model = DeepIsolationForest()
                model.fit(dif_data)
                tmp_DIF_scores = model.predict(dif_data)
                DIF_scores.extend(tmp_DIF_scores)
                del dif_data
                # sort image_list by DIF_scores in decreasing order

            idx = np.argsort(-np.array(DIF_scores))
            DIF_sorted_score_list = np.array(DIF_scores)[idx].tolist()
            DIF_results_list = np.array(image_list)[idx].tolist()
            for i in range(len(DIF_results_list)):
                DIF_results_list[i] = DIF_results_list[i].split('\\')[-1]
            self.save_as_json(DIF_results_list,
                              os.path.join(self.args.dataset,
                                           'results/' + self.args.model_name + '/DIF_results_list.json'))
            self.save_as_json(DIF_sorted_score_list,
                              os.path.join(self.args.dataset,
                                           'results/' + self.args.model_name + '/DIF_sorted_score_list.json'))
            end_time = time.time()
            DIF_time = end_time - start_time
            self.save_as_json(DIF_time,
                              os.path.join(self.args.dataset,
                                           'results/' + self.args.model_name + '/DIF_time.json'))

        return DIF_results_list, DIF_sorted_score_list, DIF_time

    def SimiFeat(self, data_s):
        SimiFeat_time = -1
        if os.path.exists(os.path.join(self.args.dataset,
                                       'results/' + self.args.model_name + '/SimiFeat_results_list.json')) and \
                os.path.exists(os.path.join(self.args.dataset,
                                            'results/' + self.args.model_name + '/SimiFeat_sorted_score_list.json')):
            SimiFeat_results_list = self.load_json(
                os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/SimiFeat_results_list.json'))
            SimiFeat_sorted_score_list = self.load_json(
                os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/SimiFeat_sorted_score_list.json'))
        else:
            start_time = time.time()
            # simulate get feature process (actually, we have already got the feature, but we need to calculate the total time)
            modelargs = torch.load(self.args.model_args)
            dataset_ = dataset(root=self.args.dataset, classes_path=self.args.class_path,
                               transform=modelargs['transform'],
                               image_size=eval(self.args.image_size), image_set=self.args.image_set,
                               specific_label=None, data_s=data_s)
            data_loader = torch.utils.data.DataLoader(dataset_, batch_size=1, shuffle=False, num_workers=0)
            model = eval(self.args.model_name)()
            model.load_state_dict(torch.load(self.args.model))
            loss_fn = modelargs['loss_fn']
            softmax_func = nn.Softmax(dim=1)
            model.to(device)
            model.eval()
            with torch.no_grad():
                for i, (images, labels, image_paths) in enumerate(data_loader):
                    out = model(images.to(device))
                    labels = labels.to(device)
                    print('\r', 'processing image: ', i, end='')
                    Loss = loss_fn(softmax_func(out), labels).cpu().numpy().item()

            org_SFM_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/org_SFM_list.json'))
            gt_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/gt_list.json'))
            image_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/image_list.json'))
            classes = self.load_json(self.args.class_path)
            class_num = len(classes.keys())
            X = np.array(org_SFM_list)
            y = np.array(gt_list)
            image_list = np.array(image_list)

            # random shuffle X,y in the same order, and record the index to recover the original order
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]
            image_list = image_list[idx]

            # split X,y,idx into parts
            X_split = np.array_split(X, 10)
            y_split = np.array_split(y, 10)

            detected_fault_numbers_summary = []

            # for each part
            for i in range(len(X_split)):
                X = X_split[i]
                y = y_split[i]

                X_torch = torch.Tensor(X)
                y_torch = torch.Tensor(y)
                clstr = compute_apparent_clusterability(X, y)
                clstr_torch = compute_apparent_clusterability_torch(X_torch, y_torch)
                diff = abs(clstr - clstr_torch)
                assert diff < 0.01, "diff should be less than 1%"
                print("Difference of two approaches: ", diff)
                detected_fault_numbers = simiFeat(21, class_num, X, y, "rank")

                detected_fault_numbers_summary.extend(detected_fault_numbers)

                torch.cuda.empty_cache()

            # recover the original order
            org_idx = np.argsort(idx)
            detected_fault_numbers_summary = np.array(detected_fault_numbers_summary)[org_idx].tolist()
            image_list = image_list[org_idx].tolist()

            # sort image_list by detected_fault_numbers_summary in decreasing order
            idx = np.argsort(-np.array(detected_fault_numbers_summary))
            SimiFeat_sorted_score_list = np.array(detected_fault_numbers_summary)[idx].tolist()
            SimiFeat_results_list = np.array(image_list)[idx].tolist()
            for i in range(len(SimiFeat_results_list)):
                SimiFeat_results_list[i] = SimiFeat_results_list[i].split('\\')[-1]
            self.save_as_json(SimiFeat_results_list,
                              os.path.join(self.args.dataset,
                                           'results/' + self.args.model_name + '/SimiFeat_results_list.json'))
            self.save_as_json(SimiFeat_sorted_score_list,
                              os.path.join(self.args.dataset,
                                           'results/' + self.args.model_name + '/SimiFeat_sorted_score_list.json'))

            end_time = time.time()
            SimiFeat_time = end_time - start_time
            self.save_as_json(SimiFeat_time,
                              os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/SimiFeat_time.json')
                              )
        return SimiFeat_results_list, SimiFeat_sorted_score_list, SimiFeat_time

    def get_loader(self, data_s, modelargs, pred, prob):
        labeled_trainset = dataset_NCNV(root=self.args.dataset, classes_path=self.args.class_path,
                                        transform=modelargs['transform'],
                                        image_size=eval(self.args.image_size), image_set=self.args.image_set,
                                        data_s=data_s, mode='labeled', pred=pred, probability=prob)
        labeled_trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=self.args.retrain_bs,
                                                          shuffle=True,
                                                          num_workers=0)
        unlabeled_trainset = dataset_NCNV(root=self.args.dataset, classes_path=self.args.class_path,
                                          transform=modelargs['transform'],
                                          image_size=eval(self.args.image_size),
                                          image_set=self.args.image_set,
                                          data_s=data_s, mode='unlabeled', pred=pred, probability=prob)
        unlabeled_trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=self.args.retrain_bs,
                                                            shuffle=True,
                                                            num_workers=0)
        return labeled_trainloader, unlabeled_trainloader

    def NCNV(self, data_s):
        NCNV_time = -1
        if os.path.exists(os.path.join(self.args.dataset,
                                       'results/' + self.args.model_name + '/NCNV_results_list.json')) and \
                os.path.exists(os.path.join(self.args.dataset,
                                            'results/' + self.args.model_name + '/NCNV_sorted_score_list.json')):
            NCNV_results_list = self.load_json(
                os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/NCNV_results_list.json'))
            NCNV_sorted_score_list = self.load_json(
                os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/NCNV_sorted_score_list.json'))
        else:
            start_time = time.time()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            epoches = 20
            threshold_sver = 0.75
            threshold_scor = 0.0
            classes = self.load_json(self.args.class_path)
            class_num = len(classes.keys())
            model1 = eval(self.args.model_name + '_NCNV')()
            model2 = eval(self.args.model_name + '_NCNV')()
            modelargs = torch.load(self.args.model_args)
            loss_fn = nn.CrossEntropyLoss()
            image_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/image_list.json'))
            if modelargs['optimizer'] == 'SGD':
                optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)
                optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)
            elif modelargs['optimizer'] == 'Adam':
                optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
                optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
            else:
                raise ValueError('optimizer not supported')
            datasets = dataset(root=self.args.dataset, classes_path=self.args.class_path,
                               transform=modelargs['transform'],
                               image_size=eval(self.args.image_size), image_set=self.args.image_set,
                               data_s=data_s)

            data_loader = torch.utils.data.DataLoader(datasets, batch_size=self.args.retrain_bs, shuffle=True,
                                                      num_workers=0)
            eval_loader = torch.utils.data.DataLoader(datasets, batch_size=self.args.retrain_bs, shuffle=False,
                                                      num_workers=0)
            dataset_name = self.args.dataset.split('/')[-1]
            test_data_s = self.data_slice(self.args, './dataset/OriginalTestData/' + dataset_name + '/test')[0]
            test_data = dataset(root='./dataset/OriginalTestData/' + dataset_name + '/',
                                classes_path=self.args.class_path,
                                transform=modelargs['transform'],
                                image_size=eval(self.args.image_size),
                                image_set='test',
                                specific_label=None,
                                ignore_list=[],
                                data_s=test_data_s)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.args.retrain_bs, shuffle=False,
                                                      num_workers=0)

            warmup = 3
            model1.to(device)
            model2.to(device)

            fea_dim = -1
            scores = None
            for epoch in range(epoches):
                model1.train()
                model2.train()
                if epoch < warmup:
                    for i, (images, labels, image_paths) in enumerate(data_loader):
                        out1, fea1 = model1(images.to(device), feat=True)
                        out2 = model2(images.to(device))
                        fea_dim = fea1.shape[1]
                        labels = labels.to(device)
                        loss1 = loss_fn(out1, labels)
                        loss2 = loss_fn(out2, labels)
                        optimizer1.zero_grad()
                        optimizer2.zero_grad()
                        loss1.backward()
                        loss2.backward()
                        optimizer1.step()
                        optimizer2.step()
                        print('\r', 'epoch: ', epoch, 'processing batch: ', i, end='')
                else:
                    print('\nfea_dim:', fea_dim)
                    prob1 = ncnv(model1, eval_loader, batch_size=self.args.retrain_bs, num_class=class_num,
                                 feat_dim=fea_dim)
                    pred1 = (prob1 < threshold_sver)
                    prob2 = ncnv(model2, eval_loader, batch_size=self.args.retrain_bs, num_class=class_num,
                                 feat_dim=fea_dim)
                    pred2 = (prob2 < threshold_sver)
                    print(pred2)
                    if sum(pred2) == 0:
                        pred2[0] = True
                    elif sum(pred2) == len(pred2):
                        pred2[0] = False
                    if sum(pred1) == 0:
                        pred1[0] = True
                    elif sum(pred1) == len(pred1):
                        pred1[0] = False

                    if epoch == epoches - 1:
                        scores = prob1

                    # train model1
                    labeled_trainloader, unlabeled_trainloader = self.get_loader(data_s, modelargs, pred2, 1 - prob2)
                    pseudo_labels = nclc(model1, model2, labeled_trainloader, unlabeled_trainloader, test_loader,
                                         batch_size=self.args.retrain_bs, num_class=class_num,
                                         threshold_scor=threshold_scor, feat_dim=fea_dim)
                    train(model1, model2, optimizer1, labeled_trainloader, unlabeled_trainloader, pseudo_labels,
                          self.args.retrain_bs,
                          class_num)

                    # train model2
                    labeled_trainloader, unlabeled_trainloader = self.get_loader(data_s, modelargs, pred1, 1 - prob1)
                    pseudo_labels = nclc(model2, model1, labeled_trainloader, unlabeled_trainloader, test_loader,
                                         batch_size=self.args.retrain_bs, num_class=class_num,
                                         threshold_scor=threshold_scor, feat_dim=fea_dim)
                    train(model2, model1, optimizer2, labeled_trainloader, unlabeled_trainloader, pseudo_labels,
                          self.args.retrain_bs,
                          class_num)
                test(epoch, model1, model2, test_loader)
            # sort image_list by scores in descending order
            idx = np.argsort(-scores)
            NCNV_sorted_score_list = scores[idx].tolist()
            NCNV_results_list = np.array(image_list)[idx].tolist()
            for i in range(len(NCNV_results_list)):
                NCNV_results_list[i] = NCNV_results_list[i].split('\\')[-1]
            self.save_as_json(NCNV_results_list, os.path.join(self.args.dataset,
                                                              'results/' + self.args.model_name + '/NCNV_results_list.json'))
            self.save_as_json(NCNV_sorted_score_list, os.path.join(self.args.dataset,
                                                                   'results/' + self.args.model_name + '/NCNV_sorted_score_list.json'))
            end_time = time.time()
            NCNV_time = end_time - start_time
            self.save_as_json(NCNV_time, os.path.join(self.args.dataset,
                                                      'results/' + self.args.model_name + '/NCNV_time.json'))
        return NCNV_results_list, NCNV_sorted_score_list, NCNV_time

    def DeepGini(self,data_s):
        if os.path.exists(os.path.join(self.args.dataset,
            'results/' + self.args.model_name + '/DeepGini_results_list.json')):
            DeepGini_results_list = self.load_json(os.path.join(self.args.dataset,
                                                                'results/' + self.args.model_name + '/DeepGini_results_list.json'))
            DeepGini_sorted_score_list = self.load_json(os.path.join(self.args.dataset,
                                                                        'results/' + self.args.model_name + '/DeepGini_sorted_score_list.json'))
            DeepGini_time = self.load_json(os.path.join(self.args.dataset,
                                                        'results/' + self.args.model_name + '/DeepGini_time.json'))
        else:
            start_time = time.time()
            # simulate get feature process (actually, we have already got the feature, but we need to calculate the total time)
            modelargs = torch.load(self.args.model_args)
            dataset_ = dataset(root=self.args.dataset, classes_path=self.args.class_path,
                               transform=modelargs['transform'],
                               image_size=eval(self.args.image_size), image_set=self.args.image_set,
                               specific_label=None, data_s=data_s)
            data_loader = torch.utils.data.DataLoader(dataset_, batch_size=1, shuffle=False, num_workers=0)
            model = eval(self.args.model_name)()
            model.load_state_dict(torch.load(self.args.model))
            loss_fn = modelargs['loss_fn']
            softmax_func = nn.Softmax(dim=1)
            model.to(device)
            model.eval()
            if self.args.model_name != 'WaveMix':
                with torch.no_grad():
                    for i, (images, labels, image_paths) in enumerate(data_loader):
                        out = model(images.to(device))
                        labels = labels.to(device)
                        print('\r', 'processing image: ', i, end='')
                        Loss = loss_fn(softmax_func(out), labels).cpu().numpy().item()
            org_SFM_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/org_SFM_list.json'))
            image_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/image_list.json'))
            classes = self.load_json(self.args.class_path)
            class_num = len(classes.keys())

            # DeepGini

            DeepGini_sorted_score_list = []
            for i in range(len(org_SFM_list)):
                tmp_gini_score = 0
                for j in range(class_num):
                    tmp_gini_score += org_SFM_list[i][j] ** 2
                tmp_gini_score = 1 - tmp_gini_score
                DeepGini_sorted_score_list.append(tmp_gini_score)
            idx = np.argsort(-np.array(DeepGini_sorted_score_list))
            DeepGini_sorted_score_list = np.array(DeepGini_sorted_score_list)[idx].tolist()
            DeepGini_results_list = np.array(image_list)[idx].tolist()
            for i in range(len(DeepGini_results_list)):
                DeepGini_results_list[i] = DeepGini_results_list[i].split('\\')[-1]
            self.save_as_json(DeepGini_results_list, os.path.join(self.args.dataset,
                                                                    'results/' + self.args.model_name + '/DeepGini_results_list.json'))
            self.save_as_json(DeepGini_sorted_score_list, os.path.join(self.args.dataset,
                                                                            'results/' + self.args.model_name + '/DeepGini_sorted_score_list.json'))
            end_time = time.time()
            DeepGini_time = end_time - start_time
            self.save_as_json(DeepGini_time, os.path.join(self.args.dataset,
                                                                'results/' + self.args.model_name + '/DeepGini_time.json'))
        return DeepGini_results_list, DeepGini_sorted_score_list, DeepGini_time


    def DeepState(self,data_s):
        if os.path.exists(os.path.join(self.args.dataset,
            'results/' + self.args.model_name + '/DeepState_results_list.json')):
            DeepState_results_list = self.load_json(os.path.join(self.args.dataset,
                                                                'results/' + self.args.model_name + '/DeepState_results_list.json'))
            DeepState_sorted_score_list = self.load_json(os.path.join(self.args.dataset,
                                                                        'results/' + self.args.model_name + '/DeepState_sorted_score_list.json'))
            DeepState_time = self.load_json(os.path.join(self.args.dataset,
                                                        'results/' + self.args.model_name + '/DeepState_time.json'))
        else:
            start_time = time.time()
            image_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/image_list.json'))
            modelargs = torch.load(self.args.model_args)
            dataset_ = dataset(root=self.args.dataset, classes_path=self.args.class_path,
                               transform=modelargs['transform'],
                               image_size=eval(self.args.image_size), image_set=self.args.image_set,
                               specific_label=None, data_s=data_s)
            data_loader = torch.utils.data.DataLoader(dataset_, batch_size=16, shuffle=False, num_workers=0)
            model = eval(self.args.model_name)()
            model.load_state_dict(torch.load(self.args.model))
            loss_fn = modelargs['loss_fn']
            softmax_func = nn.Softmax(dim=1)
            model.to(device)
            model.eval()

            DeepState_sorted_score_list = []
            with torch.no_grad():
                for i, (x, y, image_paths) in enumerate(data_loader):
                    x = x.to(device)
                    # y=y.to(device)
                    # x = model(x)
                    # _, prediction = torch.max(x, 1)
                    # s+=torch.sum(prediction == y)

                    t = model.embedding(x)
                    output, (h_n, c_n) = model.lstm(t)
                    # print(output.shape)
                    label_seq = []
                    for j in range(100):
                        out = model.fc1(output[:, j, :])
                        out = model.relu(out)
                        out = model.fc2(out)
                        _, prediction = torch.max(out, 1)
                        prediction = prediction.to("cpu")
                        label_seq.append(prediction.numpy())
                        # label_seq.append([prediction])
                    # label_seq=np.array(label_seq)
                    label_seq = np.array(label_seq)
                    m = label_seq.shape[1]

                    for ind in range(m):
                        s1 = set()
                        for j in range(100 - 1):
                            s1.add(str(label_seq[j][ind]) + str(label_seq[j + 1][ind]))
                        cr = 0.
                        sum = 0.
                        for j in range(100 - 1):
                            if label_seq[j][ind] != label_seq[j + 1][ind]:
                                cr += ((j + 1) ** 2)
                            sum += ((j + 1) ** 2)
                        DeepState_sorted_score_list.append(cr / sum)
            idx = np.argsort(-np.array(DeepState_sorted_score_list))
            DeepState_sorted_score_list = np.array(DeepState_sorted_score_list)[idx].tolist()
            DeepState_results_list = np.array(image_list)[idx].tolist()
            for i in range(len(DeepState_results_list)):
                DeepState_results_list[i] = DeepState_results_list[i].split('\\')[-1]
            self.save_as_json(DeepState_results_list, os.path.join(self.args.dataset,
                                                                    'results/' + self.args.model_name + '/DeepState_results_list.json'))
            self.save_as_json(DeepState_sorted_score_list, os.path.join(self.args.dataset,
                                                                            'results/' + self.args.model_name + '/DeepState_sorted_score_list.json'))
            end_time = time.time()
            DeepState_time = end_time - start_time
            self.save_as_json(DeepState_time, os.path.join(self.args.dataset,
                                                                'results/' + self.args.model_name + '/DeepState_time.json'))
        return DeepState_results_list, DeepState_sorted_score_list, DeepState_time


    def Uncertainty(self,data_s):
        if os.path.exists(os.path.join(self.args.dataset,
            'results/' + self.args.model_name + '/Uncertainty_results_list.json')):
            Uncertainty_results_list = self.load_json(os.path.join(self.args.dataset,
                                                                'results/' + self.args.model_name + '/Uncertainty_results_list.json'))
            Uncertainty_sorted_score_list = self.load_json(os.path.join(self.args.dataset,
                                                                        'results/' + self.args.model_name + '/Uncertainty_sorted_score_list.json'))
            Uncertainty_time = self.load_json(os.path.join(self.args.dataset,
                                                        'results/' + self.args.model_name + '/Uncertainty_time.json'))
        else:
            start_time = time.time()
            image_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/image_list.json'))
            modelargs = torch.load(self.args.model_args)
            dataset_ = dataset(root=self.args.dataset, classes_path=self.args.class_path,
                               transform=modelargs['transform'],
                               image_size=eval(self.args.image_size), image_set=self.args.image_set,
                               specific_label=None, data_s=data_s)
            data_loader = torch.utils.data.DataLoader(dataset_, batch_size=128, shuffle=False, num_workers=0)
            model = eval(self.args.model_name)()
            model.load_state_dict(torch.load(self.args.model))
            loss_fn = modelargs['loss_fn']
            softmax_func = nn.Softmax(dim=1)
            model.to(device)
            model.train()
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

            T = 20
            Uncertainty_sorted_score_list = []

            with torch.no_grad():
                for i, (X, y,_) in enumerate(data_loader):
                    output_list = []
                    for j in range(T):
                        if self.args.model_name == 'TCDCNN':
                            output_list.append(torch.unsqueeze(model(X.float().to(device)), dim=0))
                        else:
                            output_list.append(torch.unsqueeze(F.softmax(model(X.to(device)), dim=1), dim=0))
                    for j in range(X.shape[0]):
                        sub_put = []
                        for k in range(T):
                            sub_put.append(output_list[k][0][j])
                        output_variance = torch.cat(sub_put, 0).var(dim=0).mean().item()
                        Uncertainty_sorted_score_list.append(output_variance)
            idx = np.argsort(-np.array(Uncertainty_sorted_score_list))
            Uncertainty_sorted_score_list = np.array(Uncertainty_sorted_score_list)[idx].tolist()
            Uncertainty_results_list = np.array(image_list)[idx].tolist()
            if self.args.model_name != 'TCDCNN':
                for i in range(len(Uncertainty_results_list)):
                    Uncertainty_results_list[i] = Uncertainty_results_list[i].split('\\')[-1]
            self.save_as_json(Uncertainty_results_list, os.path.join(self.args.dataset,
                                                                    'results/' + self.args.model_name + '/Uncertainty_results_list.json'))
            self.save_as_json(Uncertainty_sorted_score_list, os.path.join(self.args.dataset,
                                                                            'results/' + self.args.model_name + '/Uncertainty_sorted_score_list.json'))
            end_time = time.time()
            Uncertainty_time = end_time - start_time
            self.save_as_json(Uncertainty_time, os.path.join(self.args.dataset,
                                                                'results/' + self.args.model_name + '/Uncertainty_time.json'))
        return Uncertainty_results_list, Uncertainty_sorted_score_list, Uncertainty_time




    def CleanLab(self,data_s):
        if os.path.exists(os.path.join(self.args.dataset,
            'results/' + self.args.model_name + '/CleanLab_results_list.json')):
            CleanLab_results_list = self.load_json(os.path.join(self.args.dataset,
                                                                'results/' + self.args.model_name + '/CleanLab_results_list.json'))
            CleanLab_sorted_score_list = self.load_json(os.path.join(self.args.dataset,
                                                                        'results/' + self.args.model_name + '/CleanLab_sorted_score_list.json'))
            CleanLab_time = self.load_json(os.path.join(self.args.dataset,
                                                        'results/' + self.args.model_name + '/CleanLab_time.json'))
        else:

            start_time = time.time()
            # simulate get feature process (actually, we have already got the feature, but we need to calculate the total time)
            modelargs = torch.load(self.args.model_args)
            dataset_ = dataset(root=self.args.dataset, classes_path=self.args.class_path,
                               transform=modelargs['transform'],
                               image_size=eval(self.args.image_size), image_set=self.args.image_set,
                               specific_label=None, data_s=data_s)
            data_loader = torch.utils.data.DataLoader(dataset_, batch_size=1, shuffle=False, num_workers=0)
            model = eval(self.args.model_name)()
            model.load_state_dict(torch.load(self.args.model))
            loss_fn = modelargs['loss_fn']
            softmax_func = nn.Softmax(dim=1)
            model.to(device)
            model.eval()
            if self.args.model_name != 'WaveMix':
                with torch.no_grad():
                    for i, (images, labels, image_paths) in enumerate(data_loader):
                        out = model(images.to(device))
                        labels = labels.to(device)
                        print('\r', 'processing image: ', i, end='')
                        Loss = loss_fn(softmax_func(out), labels).cpu().numpy().item()
            org_SFM_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/org_SFM_list.json'))
            image_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/image_list.json'))
            classes = self.load_json(self.args.class_path)
            class_num = len(classes.keys())
            gt_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/gt_list.json'))
            CleanLab_sorted_score_list = cleanlab.rank.get_label_quality_scores(
                labels=np.array(gt_list),
                pred_probs=np.array(org_SFM_list),
                method='self_confidence',
                adjust_pred_probs=False,
            )
            idx = np.argsort(CleanLab_sorted_score_list)
            CleanLab_sorted_score_list = np.array(CleanLab_sorted_score_list)[idx].tolist()
            CleanLab_results_list = np.array(image_list)[idx].tolist()
            for i in range(len(CleanLab_results_list)):
                CleanLab_results_list[i] = CleanLab_results_list[i].split('\\')[-1]
            self.save_as_json(CleanLab_results_list, os.path.join(self.args.dataset,
                                                                        'results/' + self.args.model_name + '/CleanLab_results_list.json'))
            self.save_as_json(CleanLab_sorted_score_list, os.path.join(self.args.dataset,
                                                                                'results/' + self.args.model_name + '/CleanLab_sorted_score_list.json'))
            end_time = time.time()
            CleanLab_time = end_time - start_time
            self.save_as_json(CleanLab_time, os.path.join(self.args.dataset,
                                                                    'results/' + self.args.model_name + '/CleanLab_time.json'))
        return CleanLab_results_list, CleanLab_sorted_score_list, CleanLab_time

    def Random(self):
        if os.path.exists(os.path.join(self.args.dataset,
            'results/' + self.args.model_name + '/Random_results_list.json')):
            Random_results_list = self.load_json(os.path.join(self.args.dataset,
                                                                'results/' + self.args.model_name + '/Random_results_list.json'))
            Random_sorted_score_list = self.load_json(os.path.join(self.args.dataset,
                                                                        'results/' + self.args.model_name + '/Random_sorted_score_list.json'))
            Random_time = self.load_json(os.path.join(self.args.dataset,
                                                        'results/' + self.args.model_name + '/Random_time.json'))
        else:
            Random_start_time = time.time()

            random.seed(2023)
            image_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/image_list.json'))

            # generate random list length of image_list range of (0,1)
            random_list = []
            for i in range(len(image_list)):
                random_list.append(random.random())
            idx = np.argsort(-np.array(random_list))
            Random_sorted_score_list = np.array(random_list)[idx].tolist()
            Random_results_list = np.array(image_list)[idx].tolist()
            if self.args.model_name != 'TCDCNN':
                for i in range(len(Random_results_list)):
                    Random_results_list[i] = Random_results_list[i].split('\\')[-1]
            self.save_as_json(Random_results_list, os.path.join(self.args.dataset,
                                                                            'results/' + self.args.model_name + '/Random_results_list.json'))
            self.save_as_json(Random_sorted_score_list, os.path.join(self.args.dataset,
                                                                            'results/' + self.args.model_name + '/Random_sorted_score_list.json'))
            Random_end_time = time.time()
            Random_time = Random_end_time - Random_start_time
            self.save_as_json(Random_time, os.path.join(self.args.dataset,
                                                                        'results/' + self.args.model_name + '/Random_time.json'))

        return Random_results_list, Random_sorted_score_list, Random_time


    def save_as_json(self, data, save_path):
        data_json = json.dumps(data, indent=4)
        with open(save_path, 'w') as file:
            file.write(data_json)

    def load_json(self, load_path):
        with open(load_path, 'r') as f:
            data = json.load(f)
        return data

    def data_slice(self, args, path_dir, slice_num=1):
        slice_num = slice_num
        random.seed(2023)
        with open(args.class_path, 'r') as f:
            classes = json.load(f)
        class_keys = list(classes.keys())
        result = {i: {} for i in range(slice_num)}
        for name in class_keys:
            img_list_dir = os.listdir(os.path.join(path_dir, name))
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
