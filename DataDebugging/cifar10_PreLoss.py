import math
import pickle
import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from train_model.models import ResNet20, VGG
from torchtext.data.utils import get_tokenizer

device = "cuda"

def PreLoss(datapath, savedatapath, model, ratio,modeltype):
    model.eval()
    trainarr = np.load(datapath, allow_pickle=True)
    print(trainarr.shape)
    newdata = []
    cnt = 0
    sum = 0
    for lb in range(10):
        tmp = []
        for i in range(trainarr.shape[0]):

            if int(trainarr[i][0]) == lb:
                tmp.append(trainarr[i])
        tmp = np.array(tmp)

        print(tmp[:, 2].sum())

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        x_train = torch.from_numpy(np.array([transform_train(x).numpy() for x in tmp[:, 1]]))
        y_train = torch.from_numpy(np.array([int(x) for x in tmp[:, 0]]))

        if modeltype == 'ResNet':
            weight_p, bias_p = [], []
            for name, p in model.named_parameters():
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]
            cost = nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay': 1e-4},
                                         {'params': bias_p, 'weight_decay': 0}], lr=0.1,
                                        momentum=0.9)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[82, 122, 163], gamma=0.1,
                                                             last_epoch=-1)
        elif modeltype == 'VGG':
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5,
                                                                   min_lr=0.000001)
            cost = nn.CrossEntropyLoss().to(device)
        res = []
        model.to(device)
        with torch.no_grad():
            for i in tqdm(range(x_train.shape[0])):
                X = x_train[i].reshape(1, 3, 32, 32)
                X = X.to(device)
                y = []
                y.append(y_train[i])
                y = np.array(y)
                y = torch.from_numpy(y)
                y = y.long()
                y = y.to(device)
                output = model(X)

                tt, pred = torch.max(output, 1)
                cur_loss = cost(output, y)
                num_correct = (pred == y).sum()
                sum += num_correct.item()

                cnt += 1

                res.append([tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], tmp[i][4], float(cur_loss)])  # 现在方法


        res = np.array(res)
        res = res[res[:, 5].argsort()[::-1]]
        sum1 = res.shape[0]
        cnt1 = 0
        for i in range(res.shape[0]):
            if i <= int(res.shape[0] * ratio):
                res[i][5] = 1
                newdata.append(res[i])
                if int(res[i][2]) == 1:
                    cnt1 += 1
            else:
                res[i][5] = 0
                newdata.append(res[i])

    np.save(savedatapath, newdata)


if __name__ == "__main__":
    modelpath = './models/cifar10_alllabel_VGG.pth'
    datapath = './data/CIFA10/CIFA10_PNG/alllabeltraindata_VAE_Kmeans_VGG.npy'
    model =VGG('VGG16')
    state_dict = torch.load(modelpath)
    model.load_state_dict(state_dict)
    PreLoss(datapath=datapath, savedatapath='', model=model, ratio=0.05,modeltype='VGG')

