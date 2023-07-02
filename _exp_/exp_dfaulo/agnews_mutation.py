import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd.grad_mode import F
from torch.utils.data import TensorDataset
from tqdm import tqdm

from myvocab import Vocab
from torchtext.data.utils import get_tokenizer
from _exp_.train_model.models import LSTM, BiLSTM

vocab = pickle.load(open("./data/AgNews/vocab.pkl", "rb"))

device = "cuda"


# [lb,context,isdirty,VAE,Kmeans,Confident]

def getlockmodelLSTM(modelpath):
    premodel = LSTM(voc_len=len(vocab), PAD=vocab.PAD)

    state_dict = torch.load(modelpath)
    premodel.load_state_dict(state_dict)
    for param in premodel.parameters():
        param.requires_grad = False

    premodel.fc1 = nn.Linear(128, 64)
    premodel.fc2 = nn.Linear(64, 4)

    return premodel


def getlockmodelBiLSTM(modelpath):
    premodel = BiLSTM(voc_len=len(vocab), PAD=vocab.PAD)

    state_dict = torch.load(modelpath)
    premodel.load_state_dict(state_dict)
    for param in premodel.parameters():
        param.requires_grad = False

    premodel.fc1 = nn.Linear(128*2, 128)
    premodel.fc2 = nn.Linear(128, 4)

    return premodel


def retrain(traindatapath, modelsavepath, model, traintype):
    trainarr = np.load(traindatapath, allow_pickle=True)

    delind = -1
    if traintype == 'VAE':
        delind = 3
    elif traintype == 'Kmeans':
        delind = 4
    elif traintype == 'Confident' or traintype == 'LOSS':
        delind = 5

    deldata = []
    for i in range(trainarr.shape[0]):
        if int(trainarr[i][delind]) == 0:
            deldata.append([trainarr[i][0], trainarr[i][1]])
    deldata = np.array(deldata)
    print('deldata shape:', deldata.shape)

    x_train = torch.from_numpy(np.array([x for x in deldata[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in deldata[:, 0]]))

    traindataset = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=256, shuffle=True)

    min_acc = 0
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)
    epoch = 1
    for t in range(epoch):
        loss, current, n = 0.0, 0.0, 0
        model.train()
        for batch, (X, y) in tqdm(enumerate(train_loader)):
            y = y.long()
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print('train_loss' + str(loss / n))
        print('train_acc' + str(current / n))
    # torch.save(model.state_dict(), modelsavepath)
    print("DONE")


if __name__ == "__main__":
    modelpath = './models/agnews_alllabel_BiLSTM.pth'
    datapath = './data/AgNews/AgNews_NEWDATA_Out/alllabeltraindata_VAE_Kmeans_LOSS_BiLSTM.npy'
    model = BiLSTM(voc_len=len(vocab), PAD=vocab.PAD)
    state_dict = torch.load(modelpath)
    model.load_state_dict(state_dict)
    start=time.time()
    retrain(datapath, '', model, 'LOSS')
    end=time.time()
    print('执行时间:',end-start)

    # arg
    # dtlist = ['alllabel', 'ranlabel', 'alldirty', 'randirty']
    # mdlist = ['LSTM', 'BiLSTM']
    # ttlist = ['VAE', 'LOSS']
    # dellist = ['003', '010', '020']
    # retraintype = 'direct'
    # ######################################################
    # for dataratio in dellist:
    #     for modeltype in mdlist:
    #         for datatype in dtlist:
    #             for traintype in ttlist:
    #                 print('now run:' + dataratio + ' ' + modeltype + ' ' + datatype + ' ' + traintype)
    #                 datapath = 'F:/ICSEdata/data/AgNews/' + datatype + 'traindata_VAE_Kmeans_LOSS_' + modeltype + dataratio + '.npy'
    #
    #                 modelpath = './models/agnews_' + datatype + '_' + modeltype + '.pth'
    #
    #                 if retraintype == 'direct':
    #                     modelsavepath = 'F:/ICSEdata/model/agnews_' + datatype + '_' + modeltype + '_retrain_' + traintype + dataratio + '.pth'
    #                 # elif retraintype=='randomweight':
    #                 #     modelsavepath = './retrainmodel/agnews_' + datatype + '_' + modeltype + '_retrain_' + traintype + '_randomweight.pth'
    #
    #                 if modeltype == 'LSTM':
    #                     if retraintype == 'direct':
    #                         model = LSTM(voc_len=len(vocab), PAD=vocab.PAD)
    #                         state_dict = torch.load(modelpath)
    #                         model.load_state_dict(state_dict)
    #                         retrain(datapath, modelsavepath, model, traintype)
    #                     # elif retraintype == 'randomweight':
    #                     #     model = getlockmodelLSTM(modelpath)
    #                     #     retrain(datapath, modelsavepath, model, traintype)
    #
    #                 elif modeltype == 'BiLSTM':
    #                     if retraintype == 'direct':
    #                         model = BiLSTM(voc_len=len(vocab), PAD=vocab.PAD)
    #                         state_dict = torch.load(modelpath)
    #                         model.load_state_dict(state_dict)
    #                         retrain(datapath, modelsavepath, model, traintype)
    #                     # elif retraintype == 'randomweight':
    #                     #     model = getlockmodelBiLSTM(modelpath)
    #                     #     retrain(datapath, modelsavepath, model, traintype)
