import math
import pickle
import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from _exp_.train_model.models import ResNet20, VGG
from torchtext.data.utils import get_tokenizer

device = "cuda"

def clusterResNet(datapath, savedatapath, model):

    trainarr = np.load(datapath, allow_pickle=True)
    print(trainarr.shape)
    newdata = []
    for lb in range(10):
        tmp = []
        for i in range(trainarr.shape[0]):
            if int(trainarr[i][0]) == lb:
                tmp.append(trainarr[i])
        tmp = np.array(tmp)
        print(tmp[:,2].sum())
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        x_train = torch.from_numpy(np.array([transform_train(x).numpy() for x in tmp[:, 1]]))
        y_train= np.array([int(x) for x in tmp[:, 2]])
        print(y_train.sum())
        ind = [x for x in range(x_train.shape[0])]
        random.shuffle(ind)
        x_train = x_train[ind]
        tmp = tmp[ind]

        res = []
        model.to(device)
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(x_train.shape[0])):
                X = x_train[i].reshape(1, 3, 32, 32)
                X = X.to(device)
                out = model.conv1(X)
                out = model.layer1(out)
                out = model.layer2(out)
                out = model.layer3(out)
                out = F.avg_pool2d(out, 8)
                out = out.view(out.size(0), -1)
                # print(out.shape)
                out = out.cpu()
                out = out.numpy()
                out = out.reshape(out.shape[1])
                # print(out.shape)
                res.append(out)
        res = np.array(res)
        print(res.shape)
        clf = KMeans(n_clusters=2)
        clf.fit(res)
        y_label = clf.labels_

        cnt = 0
        sum = 0
        for i in range(y_label.shape[0]):
            if y_label[i] == 1:
                cnt += 1
            elif y_label[i] == 0:
                sum += 1
        print(cnt, sum)

        seldata = -1
        if cnt <= sum:
            seldata = 0
            lessdata = 1
        else:
            seldata = 1
            lessdata = 0

        selnum = 0
        for i in range(tmp.shape[0]):
            if y_label[i] == seldata:
                newdata.append([tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], 0])
                selnum += 1
            else:
                newdata.append([tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], 1])
        print(selnum)

    np.save(savedatapath, newdata)

def clusterVGG(datapath, savedatapath, model):

    trainarr = np.load(datapath, allow_pickle=True)
    print(trainarr.shape)

    newdata = []
    for lb in range(10):
        tmp = []
        for i in range(trainarr.shape[0]):
            if int(trainarr[i][0]) == lb:
                tmp.append(trainarr[i])
        tmp = np.array(tmp)
        print(tmp[:,2].sum())
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        x_train = torch.from_numpy(np.array([transform_train(x).numpy() for x in tmp[:, 1]]))
        y_train= np.array([int(x) for x in tmp[:, 2]])
        print(y_train.sum())
        ind = [x for x in range(x_train.shape[0])]
        random.shuffle(ind)
        x_train = x_train[ind]
        tmp = tmp[ind]

        res = []
        model.to(device)
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(x_train.shape[0])):
                X = x_train[i].reshape(1, 3, 32, 32)
                X = X.to(device)
                out = model.features(X)
                out = out.cpu()
                out = out.numpy()
                out = out.reshape(out.shape[1])

                res.append(out)
        res = np.array(res)
        print(res.shape)
        clf = KMeans(n_clusters=2)
        clf.fit(res)
        y_label = clf.labels_
        cnt = 0
        sum = 0
        for i in range(y_label.shape[0]):
            if y_label[i] == 1:
                cnt += 1
            elif y_label[i] == 0:
                sum += 1
        print(cnt, sum)

        seldata = -1
        if cnt <= sum:
            seldata = 0
            lessdata = 1
        else:
            seldata = 1
            lessdata = 0

        selnum = 0
        for i in range(tmp.shape[0]):
            if y_label[i] == seldata:
                newdata.append([tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], 0])
                selnum += 1
            else:
                newdata.append([tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], 1])
        print(selnum)


    np.save(savedatapath, newdata)

if __name__ == "__main__":
    model = VGG('VGG16')
    modelpath = './models/cifar10_alllabel_VGG.pth'
    datapath = './data/CIFA10/CIFA10_PNG/alllabeltraindata_VAE.npy'
    state_dict = torch.load(modelpath)
    model.load_state_dict(state_dict)

    clusterVGG(datapath, '', model)


