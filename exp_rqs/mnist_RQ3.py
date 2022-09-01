import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import xlwt
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from train_model.models import LeNet5, LeNet1

device = 'cuda'


def val(dataloader, model, loss_fn):
    # 将模型转为验证模式
    model.eval()
    model.to(device)
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
        # print('val_loss' + str(loss / n))
        # print('val_acc' + str(current / n))
        return current / n


def RQ3(datatype, modeltype, baseline, split_ratio):
    if baseline == 'offline' or baseline == 'online':
        data = np.load('F:\\ICSEdata\\online_new\\MNIST\\' + datatype +'_'+ modeltype + '_' + baseline + '.npy', allow_pickle=True)
    elif baseline == 'entire':
        data = np.load('F:/ICSEdata/RQ1data/MNIST/' + datatype + modeltype + '_online.npy', allow_pickle=True)

    if modeltype == 'LeNet1':
        model = LeNet1()
    if modeltype == 'LeNet5':
        model = LeNet5()
    if baseline == 'offline' or baseline == 'online':
        state_dict = torch.load('./models/mnist_' + datatype + '_' + modeltype + '.pth')
        model.load_state_dict(state_dict)

    # load orgtraindata
    orgtraindatapath = './data/MNIST/MNIST_PNG/orgtraindata.npy'
    trainarr = np.load(orgtraindatapath, allow_pickle=True)
    x_orgtrain = torch.from_numpy(np.array([x / 255. for x in trainarr[:, 1]]))
    y_orgtrain = torch.from_numpy(np.array([int(x) for x in trainarr[:, 0]]))
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
    ])
    x_orgtrain = data_transform(x_orgtrain.reshape(x_orgtrain.shape[0], 3, 28, 28))
    orgtraindataset = TensorDataset(x_orgtrain, y_orgtrain)
    orgtrain_loader = torch.utils.data.DataLoader(dataset=orgtraindataset, batch_size=16, shuffle=True)

    # load orgtestdata
    orgtestdatapath = './data/MNIST/MNIST_PNG/orgtestdata.npy'
    testarr = np.load(orgtestdatapath, allow_pickle=True)
    x_orgtest = torch.from_numpy(np.array([x / 255. for x in testarr[:, 1]]))
    y_orgtest = torch.from_numpy(np.array([int(x) for x in testarr[:, 0]]))
    x_orgtest = data_transform(x_orgtest.reshape(x_orgtest.shape[0], 3, 28, 28))
    orgtestdataset = TensorDataset(x_orgtest, y_orgtest)
    orgtest_loader = torch.utils.data.DataLoader(dataset=orgtestdataset, batch_size=16, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失

    # print('TestAcc-Official:')
    orgtrainacc = val(orgtrain_loader, model, loss_fn)
    orgtestacc = val(orgtest_loader, model, loss_fn)

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
    # print(newdata.shape)
    if baseline == 'offline' or baseline == 'online':
        epoch = 10
    elif baseline == 'entire':
        epoch = 70
    lr = 0.001
    x_train = torch.from_numpy(np.array([x / 255. for x in newdata[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in newdata[:, 0]]))
    x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))
    traindataset = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.to(device)
    res1 = -1
    res2 = -1
    for t in tqdm(range(epoch)):

        loss, current, n = 0.0, 0.0, 0
        for batch, (X, y) in enumerate(train_loader):
            y = y.long()
            X, y = X.to(device), y.to(device)
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

        nowtestacc = val(orgtest_loader, model, loss_fn)
        if nowtestacc > res2:
            res2 = nowtestacc
            res1 = val(orgtrain_loader, model, loss_fn)

        model.train()
        # print(f"epoch{t + 1} loss{testacc}\n-------------------")
        if t == epoch - 1:
            pass
            # torch.save(model.state_dict(), modelsave_path)
    # print("DONE")
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
    dtlist = ['alllabel', 'ranlabel', 'alldirty', 'randirty']
    datasetname = 'MNIST'
    mdlist = ['LeNet1','LeNet5']
    bllist = ['online']
    splist = [0.05]
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet('1')  # 在工作簿中新建一个表格
    row = -1
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
                    workbook.save('F:/ICSEdata/excel/' + datasetname + 'newOnline_RQ3.xls')


