import pickle
import random

import numpy as np
import pandas as pd
import torch
import xlwt
from torch import nn
from torch.autograd.grad_mode import F
from torch.utils.data import TensorDataset
from tqdm import tqdm

from myvocab import Vocab
from torchtext.data.utils import get_tokenizer
from train_model.models import LSTM, BiLSTM

vocab = pickle.load(open("./data/AgNews/vocab.pkl", "rb"))

device = "cuda"


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


def val(model, dataloader):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    model.to(device)
    # print(torch.no_grad)
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            y = y.long()
            X, y = X.to(device), y.to(device)
            output = model(X)
            # cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            # loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        # print('val_loss' + str(loss / n))
        # print('val_acc' + str(current / n))

        return current / n


def RQ3(datatype, modeltype, baseline, split_ratio):
    if baseline == 'offline':
        data = np.load('F:/ICSEdata/RQ1data/AgNews/' + datatype + modeltype + '_offline.npy', allow_pickle=True)
    elif baseline == 'online':
        data = np.load('F:\\ICSEdata\\online_new\\AgNews\\' + datatype + '_' + modeltype + '_' + baseline + '.npy',
                       allow_pickle=True)
    elif baseline == 'entire':
        data = np.load('F:\\ICSEdata\\online_new\\AgNews\\' + datatype + '_' + modeltype + '_online.npy',
                       allow_pickle=True)

    if modeltype == 'LSTM':
        model = LSTM(voc_len=len(vocab), PAD=vocab.PAD)
    if modeltype == 'BiLSTM':
        model = BiLSTM(voc_len=len(vocab), PAD=vocab.PAD)

    if baseline == 'offline' or baseline == 'online':
        state_dict = torch.load('./models/agnews_' + datatype + '_' + modeltype + '.pth')
        model.load_state_dict(state_dict)

    trainarr, testarr = load_data('./data/AgNews/orgtraindata.npy', './data/AgNews/orgtestdata.npy')

    # load orgtraindata
    x_orgtrain = torch.from_numpy(np.array([x for x in trainarr[:, 1]]))
    y_orgtrain = torch.from_numpy(np.array([int(x) for x in trainarr[:, 0]]))
    orgtraindataset = TensorDataset(x_orgtrain, y_orgtrain)
    orgtrain_loader = torch.utils.data.DataLoader(dataset=orgtraindataset, batch_size=256, shuffle=True)

    # load orgtestdata
    x_orgtest = torch.from_numpy(np.array([x for x in testarr[:, 1]]))
    y_orgtest = torch.from_numpy(np.array([int(x) for x in testarr[:, 0]]))
    orgtestdataset = TensorDataset(x_orgtest, y_orgtest)
    orgtest_loader = torch.utils.data.DataLoader(dataset=orgtestdataset, batch_size=256, shuffle=True)

    orgtrainacc = val(model, orgtrain_loader)
    orgtestacc = val(model, orgtest_loader)

    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = torch.optim.Adam(model.parameters())

    if baseline == 'offline' or baseline == 'online':
        epochs = 8
    elif baseline == 'entire':
        epochs = 25

    newdata = []
    sum = 0
    for i in range(data.shape[0]):
        if i <= int(data.shape[0] * split_ratio):
            if int(data[i][2]) == 1:
                sum += 1
                if datatype == 'alllabel' or datatype == 'ranlabel':
                    newdata.append([data[i][3], data[i][1]])
                elif datatype == 'alldirty' or datatype == 'randirty':
                    pass
            elif int(data[i][2]) == 0:
                newdata.append([data[i][0], data[i][1]])
        else:
            newdata.append([data[i][0], data[i][1]])

    newdata = np.array(newdata)
    x_train = torch.from_numpy(np.array([x for x in newdata[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in newdata[:, 0]]))
    traindataset = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=256, shuffle=True)

    res1 = -1
    res2 = -1
    model.to(device)
    for t in tqdm(range(epochs)):
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
        # print('train_loss' + str(loss / n))
        # print('train_acc' + str(current / n))
        nowtrainacc = val(model, orgtrain_loader)
        nowtestacc = val(model, orgtest_loader)

        res1 = max(res1, nowtrainacc)
        res2 = max(res2, nowtestacc)

    return orgtrainacc, orgtestacc, res1, res2


def writexcel(sheet, x, P, val, color, offset):
    style1 = "font:colour_index red;"
    style2 = "font:colour_index blue;"
    style3 = "font:colour_index green;"
    style4 = "font:colour_index black;"
    if color == 'W':
        style = xlwt.easyxf(style1)
    elif color == 'T':
        style = xlwt.easyxf(style2)
    elif color == 'L':
        style = xlwt.easyxf(style3)
    else:
        style = xlwt.easyxf(style4)
    y = -1
    if P == 'input':
        y = 0
    elif P == 'hidden':
        y = 1
    elif P == 'output':
        y = 2
    elif P == 'input+hidden':
        y = 3

    elif P == 'hidden+output':
        y = 4
    elif P == 'input+output':
        y = 5
    elif P == 'all':
        y = 6
    else:
        y = P
    sheet.write(x, y + offset, val, style)


if __name__ == "__main__":
    dtlist = ['randirty']
    datasetname = 'AgNews'
    mdlist = ['BiLSTM']
    bllist = ['offline', 'online', 'entire']
    splist = [0.05]
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet('1')  # 在工作簿中新建一个表格
    row = -1
    checkpoint = 1 + 6
    for modeltype in mdlist:
        for datatype in dtlist:
            row += 1
            col = -1
            for split_ratio in splist:
                for baseline in bllist:
                    print('nowrun:' + ' ' + modeltype + ' ' + datatype + ' ' + str(split_ratio) + ' ' + baseline)
                    col += 2
                    random.seed(6657)
                    orgtrainacc, orgtestacc, res1, res2 = RQ3(datatype, modeltype, baseline, split_ratio)
                    if col == 1:
                        writexcel(sheet, row, col - 1, orgtrainacc, '', 0)
                        writexcel(sheet, row, col, orgtestacc, '', 0)
                    writexcel(sheet, row, col + 1, res1, '', 0)
                    writexcel(sheet, row, col + 2, res2, '', 0)
                    workbook.save('F:/ICSEdata/excel/' + datasetname + '_RQ3' + 'newOnlinerara' + str(checkpoint) + '.xls')
