import argparse
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd.grad_mode import F
from torch.utils.data import TensorDataset

from myvocab import Vocab
from torchtext.data.utils import get_tokenizer
from models import LSTM, BiLSTM

parser = argparse.ArgumentParser()

parser.add_argument('--vocabpath', type=str, default='vocab.pkl', help='vocab path.')
parser.add_argument('--traindata', type=str, default='./data/AgNews/orgtraindata.npy', help='traindatapath.')
parser.add_argument('--testdata', type=str, default='./data/AgNews/orgtestdata.npy', help='testdatapath.')
parser.add_argument('--modelsavepath', type=str, default='./models/agnews_org_LSTM.pth', help='modelsavepath.')
args = parser.parse_args()
device = 'cuda'
vocab = pickle.load(open(args.vocabpath, "rb"))
model = LSTM(voc_len=len(vocab), PAD=vocab.PAD)




def load_data(traindata, testdata):
    trainarr = np.load(traindata)
    testarr = np.load(testdata)
    tokenizer = get_tokenizer('basic_english')
    traind = []
    for i in range(trainarr.shape[0]):
        traind.append(
            [trainarr[i][0], vocab.transform(sentence=tokenizer(trainarr[i][1]), max_len=100), trainarr[i][2]])
    traind = np.array(traind, dtype=object)
    testd = []
    for i in range(testarr.shape[0]):
        testd.append([testarr[i][0], vocab.transform(sentence=tokenizer(testarr[i][1]), max_len=100), testarr[i][2]])
    testd = np.array(testd, dtype=object)
    return traind, testd


def val(dataloader, loss_fn):
    # 将模型转为验证模式
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    # 非训练，推理期用到（测试时模型参数不用更新， 所以no_grad）
    # print(torch.no_grad)
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            y = y.long()
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print('val_loss' + str(loss / n))
        print('val_acc' + str(current / n))

        return current / n


def acc_in_orgtestset(loss_fn):
    print("model in org datasrt:\n")
    datapath = './data/AgNews/orgtestdata.npy'
    tokenizer = get_tokenizer('basic_english')
    testarr = np.load(datapath)
    testd = []
    for i in range(testarr.shape[0]):
        testd.append([testarr[i][0], vocab.transform(sentence=tokenizer(testarr[i][1]), max_len=100), testarr[i][2]])
    testd = np.array(testd, dtype=object)
    testarr = testd
    x_test = torch.from_numpy(np.array([x for x in testarr[:, 1]]))
    y_test = torch.from_numpy(np.array([int(x) for x in testarr[:, 0]]))
    testdataset = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=256, shuffle=True)
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_loader):
            y = y.long()
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print('val_loss' + str(loss / n))
        print('val_acc' + str(current / n))

        return current / n


def train(traindata, testdata, modelsavepath):
    trainarr, testarr = load_data(traindata, testdata)
    x_train = torch.from_numpy(np.array([x for x in trainarr[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in trainarr[:, 0]]))
    x_test = torch.from_numpy(np.array([x for x in testarr[:, 1]]))
    y_test = torch.from_numpy(np.array([int(x) for x in testarr[:, 0]]))
    traindataset = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=256, shuffle=True)
    testdataset = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=256, shuffle=True)
    min_acc = 0
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)
    epoch = 25
    res1 = -1
    res2 = -1
    for t in range(epoch):
        loss, current, n = 0.0, 0.0, 0
        model.train()
        for batch, (X, y) in enumerate(train_loader):
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
        a = val(test_loader, loss_fn)
        if a > res1:
            res1 = a
            res2 = current / n
            torch.save(model.state_dict(), modelsavepath)
        print('now max train:' + ' ' + str(res2) + ' test: ' + str(res1))
        print(f"epoch{t + 1} loss{a}\n-------------------")
        if t == epoch - 1:
            torch.save(model.state_dict(), modelsavepath)
            acc_in_orgtestset(loss_fn)
    print("DONE")
if __name__ == "__main__":
    train(args.traindata, args.testdata, args.modelsavepath)