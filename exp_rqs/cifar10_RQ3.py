import random

import numpy as np
import torch
import torchvision
import xlwt
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from train_model.models import ResNet20, VGG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Learning_rate = 0.1


def val(model, test_loader):
    model.eval()
    model.to(device)
    testing_correct = 0
    total = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            y = y.long()
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, pred = torch.max(outputs, 1)
            total += y.size(0)
            testing_correct += (pred == y).sum().item()
    # print('Test acc: {:.4}%'.format(testing_correct / total))
    return testing_correct / total


def RQ3(datatype, modeltype, baseline, split_ratio):
    if baseline == 'offline' or baseline == 'online':
        data = np.load('F:\\ICSEdata\\online_new\\CIFAR10\\' + datatype +'_'+ modeltype + '_' + baseline + '.npy',
                       allow_pickle=True)
    elif baseline == 'entire':
        data = np.load('F:/ICSEdata/RQ1data/CIFAR10/' + datatype + modeltype + '_online.npy', allow_pickle=True)

    if modeltype == 'ResNet':
        model = ResNet20()
    if modeltype == 'VGG':
        model = VGG('VGG16')

    if baseline == 'offline' or baseline == 'online':
        state_dict = torch.load('./models/cifar10_' + datatype + '_' + modeltype + '.pth')
        model.load_state_dict(state_dict)
    transform = transforms.Compose([  # 测试集同样进行图像预处理
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # load orgtraindata
    orgtraindatapath = './data/CIFA10/CIFA10_PNG/orgtraindata.npy'
    trainarr = np.load(orgtraindatapath, allow_pickle=True)
    x_orgtrain = torch.from_numpy(np.array([transform(x).numpy() for x in trainarr[:, 1]]))
    y_orgtrain = torch.from_numpy(np.array([int(x) for x in trainarr[:, 0]]))
    orgtraindataset = TensorDataset(x_orgtrain, y_orgtrain)
    orgtrain_loader = torch.utils.data.DataLoader(dataset=orgtraindataset, batch_size=128, shuffle=True)

    # load orgtestdata
    orgtestdatapath = './data/CIFA10/CIFA10_PNG/orgtestdata.npy'
    testarr = np.load(orgtestdatapath, allow_pickle=True)
    x_orgtest = torch.from_numpy(np.array([transform(x).numpy() for x in testarr[:, 1]]))
    y_orgtest = torch.from_numpy(np.array([int(x) for x in testarr[:, 0]]))
    orgtestdataset = TensorDataset(x_orgtest, y_orgtest)
    orgtest_loader = torch.utils.data.DataLoader(dataset=orgtestdataset, batch_size=128, shuffle=True)

    orgtrainacc = val(model, orgtrain_loader)
    orgtestacc = val(model, orgtest_loader)

    if modeltype == 'ResNet':
        # 取出权重参数和偏置参数，仅对权重参数加惩罚系数
        weight_p, bias_p = [], []
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        cost = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate, momentum=0.9)
        optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay': 1e-4},
                                     {'params': bias_p, 'weight_decay': 0}], lr=Learning_rate,
                                    momentum=0.9)  # 内置的SGD是L2正则，且对所有参数添加惩罚，对偏置加正则易导致欠拟合，一般只对权重正则化
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[82, 122, 163], gamma=0.1,
                                                         last_epoch=-1)  # Q4: 对应多少步,epoch= 32000/(50000/batch_size),48000,64000
    elif modeltype == 'VGG':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)  # VGG16
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5,
                                                               min_lr=0.000001)  # 动态更新学习率VGG16
        cost = nn.CrossEntropyLoss().to(device)

    if baseline == 'offline' or baseline == 'online':
        epochs = 5
    elif baseline == 'entire':
        if modeltype == 'ResNet':
            epochs = 20
        elif modeltype == 'VGG':
            epochs = 20

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
    print(sum / 500)
    # print(newdata.shape)
    x_train = torch.from_numpy(np.array([transform(x).numpy() for x in newdata[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in newdata[:, 0]]))
    traindataset = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=128, shuffle=True)

    model.to(device)
    res1 = -1
    res2 = -1
    for epoch in tqdm(range(epochs)):
        model.train()

        training_loss = 0.0
        training_correct = 0
        training_acc = 0.0
        # print("Epoch {}/{}".format(epoch + 1, epochs))
        # print("-" * 30)

        total_train = 0
        for i, (X, y) in enumerate(train_loader):
            y = y.long()
            X, y = X.to(device), y.to(device)

            # print(x.shape)
            # print(label.shape)
            # 前向传播计算损失
            outputs = model(X)
            loss = cost(outputs, y)
            training_loss += loss.item()
            # print(outputs)
            _, pred = torch.max(outputs, 1)  # 预测最大值所在位置标签
            total_train += y.size(0)
            num_correct = (pred == y).sum()
            training_acc += num_correct.item()

            # 反向传播+优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('Epoch：', epoch, 'train loss:', training_loss/len(data_loader_train))
            # Loss_list.append(training_loss/len(data_loader_train))
            # if i % 100 == 99:
            #     print('[%d, %5d] traing_loss: %f' % (epoch + 1, i + 1, training_loss / 100))
            #     Loss_list.append(training_loss / 100)
            #     training_loss = 0.0

        # print('Train acc:{:.4}%'.format(training_acc / total_train))
        if modeltype == 'VGG':
            scheduler.step(training_loss / len(train_loader))
        elif modeltype == 'ResNet':
            scheduler.step()

        nowtestacc = val(model, orgtest_loader)

        if nowtestacc > res2:
            res2 = nowtestacc
            res1 = val(model, orgtrain_loader)

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
    datasetname = 'CIFAR10'
    mdlist = ['ResNet','VGG']
    bllist = ['online']
    splist = [0.05]
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet('1')  # 在工作簿中新建一个表格
    row = -1
    checkpoint = 0
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
                    workbook.save(
                        'F:/ICSEdata/excel/' + datasetname + '_RQ3' + '_newOnline' + str(checkpoint) + 'best.xls')

    # workbook.save('F:/ICSEdata/excel/' + datasetname + '_RQ3.xls')
