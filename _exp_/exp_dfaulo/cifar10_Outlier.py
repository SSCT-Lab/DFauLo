import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from pyod.models.vae import VAE
from torchvision import transforms

def Outlier(datapath, savedatapath,ratio):
    dataprg = np.load(datapath, allow_pickle=True)
    newdata = []

    for lb in range(10):
        tmp = []
        for i in range(dataprg.shape[0]):
            if int(dataprg[i][0]) == lb:
                tmp.append(dataprg[i])
        tmp = np.array(tmp)

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        x_train = torch.from_numpy(np.array([transform_train(x).numpy() for x in tmp[:, 1]]))
        print(x_train.shape)
        x_train = x_train.reshape(x_train.shape[0], 3 * 32 * 32)


        print(tmp[:, 2].sum())
        ind = [x for x in range(x_train.shape[0])]
        random.shuffle(ind)
        x_train=x_train[ind]
        tmp=tmp[ind]

        print(tmp[:, 2].sum())


        clf = VAE(epochs=5)
        clf.fit(x_train)
        y_score = clf.decision_scores_
        y_label = clf.labels_
        y_label = np.array(y_label)
        y_score = np.array(y_score)

        res = []
        for i in range(tmp.shape[0]):
            res.append([tmp[i][0], tmp[i][1], tmp[i][2], y_score[i]])
        res = np.array(res, dtype=object)


        cnt = 0
        res = res[res[:, 3].argsort()[::-1]]
        sum = res.shape[0]


        for i in range(res.shape[0]):
            if i <= int(res.shape[0] * ratio):
                res[i][3] = 1
                newdata.append(res[i])
            else:
                res[i][3] = 0
                newdata.append(res[i])

    np.save(savedatapath, newdata)


if __name__ == "__main__":

    Outlier(datapath='./data/CIFA10/CIFA10_PNG/alllabeltraindata.npy', savedatapath='', ratio=0.05)
