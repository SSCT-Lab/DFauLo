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
        for i in range(60000):
            if int(dataprg[i][0]) == lb:
                tmp.append(dataprg[i])
        tmp = np.array(tmp)
        x_train = torch.from_numpy(np.array([x / 255. for x in tmp[:, 1]]))

        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
        ])
        x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))
        x_train = x_train.reshape(x_train.shape[0], 28 * 28)
        clf = VAE(epochs=5)
        clf.fit(x_train)
        y_score = clf.decision_scores_
        y_score = np.array(y_score)
        res = []
        for i in range(tmp.shape[0]):
            res.append([tmp[i][0], tmp[i][1], tmp[i][2], y_score[i]])
        res = np.array(res,dtype=object)

        res = res[res[:, 3].argsort()[::-1]]

        res = res[int(res.shape[0] * ratio):]
        for i in range(res.shape[0]):
            newdata.append(res[i])


    newdata=np.array(newdata)
    print(newdata.shape)
    np.save(savedatapath,newdata)

if __name__ == "__main__":

    datapath = './data/MNIST/MNIST_PNG/alllabeltraindata.npy'

    Outlier(datapath=datapath,savedatapath='',ratio=0.05)
