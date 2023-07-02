import math
import pickle
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torchvision import transforms
from tqdm import tqdm

from _exp_.train_model.models import LSTM, BiLSTM
from torchtext.data.utils import get_tokenizer

device = "cuda"



def cluster(datapath, savedatapath, model):
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

        res = []
        model.to(device)
        with torch.no_grad():
            for i in tqdm(range(x_train.shape[0])):
                X = x_train[i].reshape(1, -1)
                X = X.to(device)
                embd = model.embedding(X)
                output, (h_n, c_n) = model.lstm(embd)
                out = model.fc1(output[:, -1, :])
                out = model.relu(out)
                out = out.cpu()
                out = out.numpy()
                out = out.reshape(out.shape[1])
                res.append(out)

        res = np.array(res)
        print(res.shape)
        clf = KMeans(n_clusters=2)
        clf.fit(res)
        y_label = clf.labels_
        centers = clf.cluster_centers_
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


vocab = pickle.load(open("vocab.pkl", "rb"))
if __name__ == "__main__":
    datapath = './data/AgNews/AgNews_NEWDATA_Sin/alllabeltraindata_VAE.npy'
    modelpath = './models/agnews_alllabel_BiLSTM.pth'
    model = BiLSTM(voc_len=len(vocab), PAD=vocab.PAD)
    state_dict = torch.load(modelpath)
    model.load_state_dict(state_dict)
    cluster(datapath, '', model)

