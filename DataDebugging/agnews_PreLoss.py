import pickle
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from train_model.models import LSTM, BiLSTM
from torchtext.data.utils import get_tokenizer

device = "cuda"

def PreLoss(datapath, savedatapath, model, ratio):
    trainarr = np.load(datapath, allow_pickle=True)
    print(trainarr.shape)

    newdata = []
    for lb in range(4):
        tmp = []
        for i in range(trainarr.shape[0]):
            if int(trainarr[i][0]) == lb:
                tmp.append(trainarr[i])
        tmp = np.array(tmp)

        x_train = torch.from_numpy(np.array([x for x in tmp[:, 1]]))
        y_train = torch.from_numpy(np.array([int(x) for x in tmp[:, 0]]))
        loss_fn = nn.CrossEntropyLoss()
        res = []
        model.to(device)
        with torch.no_grad():
            for i in tqdm(range(x_train.shape[0])):
                X = x_train[i].reshape(1, -1)
                X = X.to(device)
                y = []
                y.append(y_train[i])
                y = np.array(y)
                y = torch.from_numpy(y)
                y = y.long()
                y = y.to(device)
                output = model(X)
                softmax_func = nn.Softmax(dim=1)
                output = softmax_func(output)
                tt, pred = torch.max(output, axis=1)
                cur_loss = loss_fn(output, y)

                res.append([tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], tmp[i][4], float(cur_loss)])  # 现在方法


        res = np.array(res)
        res = res[res[:, 5].argsort()[::-1]]

        sum = res.shape[0]
        cnt = 0
        for i in range(res.shape[0]):
            if i <= int(res.shape[0] * ratio):
                res[i][5] = 1
                newdata.append(res[i])
                if int(res[i][2]) == 1:
                    cnt += 1
            else:
                res[i][5] = 0
                newdata.append(res[i])

    np.save(savedatapath, newdata)


vocab = pickle.load(open("vocab.pkl", "rb"))
if __name__ == "__main__":
    modelpath = './models/agnews_alllabel_BiLSTM.pth'
    datapath = './data/AgNews/AgNews_NEWDATA_Mid/alllabeltraindata_VAE_Kmeans_BiLSTM.npy'
    model = BiLSTM(voc_len=len(vocab), PAD=vocab.PAD)
    state_dict = torch.load(modelpath)
    model.load_state_dict(state_dict)
    PreLoss(datapath, '', model, 0.05)

