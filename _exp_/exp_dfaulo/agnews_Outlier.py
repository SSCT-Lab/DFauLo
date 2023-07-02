import pickle
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from pyod.models.iforest import IForest
from pyod.models.vae import VAE
from torchvision import transforms
from torch.utils.data import TensorDataset

from myvocab import Vocab
from torchtext.data.utils import get_tokenizer
from _exp_.train_model.models import LSTM, BiLSTM

vocab = pickle.load(open("vocab.pkl", "rb"))


def load_data(traindata):
    trainarr = np.load(traindata)
    tokenizer = get_tokenizer('basic_english')
    traind = []
    for i in range(trainarr.shape[0]):
        traind.append(
            [trainarr[i][0], vocab.transform(sentence=tokenizer(trainarr[i][1]), max_len=100), trainarr[i][2]])
    traind = np.array(traind, dtype=object)
    return traind


def Outlier(datapath, savedatapath, ratio):
    trainarr = load_data(datapath)

    print(trainarr.shape)
    newdata = []

    for lb in range(4):
        tmp = []
        for i in range(trainarr.shape[0]):

            if int(trainarr[i][0]) == lb:
                tmp.append(trainarr[i])

        tmp = np.array(tmp)

        x_train = torch.from_numpy(np.array([x for x in tmp[:, 1]]))

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
                if int(res[i][2]) == 1:
                    cnt += 1
            else:
                res[i][3] = 0
                newdata.append(res[i])

    np.save(savedatapath, newdata)


if __name__ == "__main__":
    datapath = './data/AgNews/alllabeltraindata.npy'
    Outlier(datapath=datapath, savedatapath='', ratio=0.05)

