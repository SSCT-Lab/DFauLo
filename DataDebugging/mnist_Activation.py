import math
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torchvision import transforms

from train_model.models import LeNet1, LeNet5

device = 'cuda'


def clusterLeNet1(datapath, savedatapath, model):
    model.to(device)
    dataprg = np.load(datapath, allow_pickle=True)
    newdata = []
    for lb in range(10):
        tmp = []
        for i in range(60000):
            if int(dataprg[i][0]) == lb:
                tmp.append(dataprg[i])
        tmp = np.array(tmp)
        x_train = torch.from_numpy(np.array([x / 255. for x in tmp[:, 1]]))

        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
        ])
        x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))

        res = []
        with torch.no_grad():
            for i in range(x_train.shape[0]):
                out = model.c1(x_train[i].reshape(1, 1, 28, 28).to(device))
                out = model.TANH(out)
                out = model.s2(out)
                out = model.c3(out)
                out = model.TANH(out)
                out = model.s4(out)
                out = out.view(out.size(0), -1)
                out = out.cpu().numpy()
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
        else:
            seldata = 1

        selnum = 0
        for i in range(tmp.shape[0]):
            if y_label[i] == seldata:
                newdata.append([tmp[i][0], tmp[i][1], tmp[i][2]])
                selnum += 1
        print(selnum)
    newdata = np.array(newdata, dtype=object)
    np.save(savedatapath, newdata)


def clusterLeNet5(datapath, savedatapath, model):
    dataprg = np.load(datapath, allow_pickle=True)
    newdata = []
    for lb in range(10):
        tmp = []
        for i in range(60000):
            if int(dataprg[i][0]) == lb:
                tmp.append(dataprg[i])
        tmp = np.array(tmp)
        x_train = torch.from_numpy(np.array([x / 255. for x in tmp[:, 1]]))

        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
        ])
        x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))

        res = []
        with torch.no_grad():
            for i in range(x_train.shape[0]):
                out = model.c1(x_train[i].reshape(1, 1, 28, 28))
                out = model.Sigmoid(out)
                out = model.s2(out)
                out = model.c3(out)
                out = model.Sigmoid(out)
                out = model.s4(out)
                out = model.c5(out)
                out = model.flatten(out)
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
        else:
            seldata = 1

        selnum = 0
        for i in range(tmp.shape[0]):
            if y_label[i] == seldata:
                newdata.append([tmp[i][0], tmp[i][1], tmp[i][2]])
                selnum += 1
        print(selnum)
    newdata = np.array(newdata, dtype=object)
    np.save(savedatapath, newdata)


if __name__ == "__main__":
    datapath = './data/MNIST/MNIST_PNG/alllabeltraindata.npy'
    modelpath = './models/mnist_alllabel_LeNet5.pth'
    savedatapath = './data/MNIST/MNIST_PNG/alllabeltraindata_Activation.npy'

    model = LeNet5()
    state_dict = torch.load(modelpath)
    model.load_state_dict(state_dict)
    clusterLeNet5(datapath, savedatapath, model)
