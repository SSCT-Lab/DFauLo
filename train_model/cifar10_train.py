import argparse
import time

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from models import ResNet20, VGG
device='cuda'
parser = argparse.ArgumentParser()


parser.add_argument('--lr', type=float, default=0.1, help='Learning Rate')
parser.add_argument('--datatype', type=str, default='alllabel', help='alllabel indicate RandomLabelNoise.'
                                                                     'ranlabel indicate SpecificLabelNoise.'
                                                                     'alldirty indicate RandomDataNoise.'
                                                                     'randirty indicate SpecificDataNoise.')
parser.add_argument('--modeltype', type=str, default='ResNet', help='ResNet or VGG')
parser.add_argument('--model_savepath', type=str, default='./models/cifar10_alllabel.pth', help='model save path')
parser.add_argument('--epochs', type=int, default=45, help='epochs')
args = parser.parse_args()


def train(datatype, epochs,modeltype, model,model_savepath):
    trainarr = np.load('data/CIFA10/CIFA10_PNG/' + datatype + 'traindata.npy', allow_pickle=True)
    testarr = np.load('data/CIFA10/CIFA10_PNG/' + datatype + 'testdata.npy', allow_pickle=True)

    transform_train = transforms.Compose([
        transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 用平均值和标准偏差归一化张量图像，

    ])

    transform_test = transforms.Compose([  # 测试集同样进行图像预处理
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    x_train = torch.from_numpy(np.array([transform_train(x).numpy() for x in trainarr[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in trainarr[:, 0]]))

    x_test = torch.from_numpy(np.array([transform_test(x).numpy() for x in testarr[:, 1]]))
    y_test = torch.from_numpy(np.array([int(x) for x in testarr[:, 0]]))

    traindataset = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=128, shuffle=True)

    testdataset = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=128, shuffle=True)




    if modeltype == 'ResNet':

        weight_p, bias_p = [], []
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        cost = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay': 1e-4},
                                     {'params': bias_p, 'weight_decay': 0}], lr=args.lr,
                                    momentum=0.9)  # 内置的SGD是L2正则，且对所有参数添加惩罚，对偏置加正则易导致欠拟合，一般只对权重正则化
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[82, 122, 163], gamma=0.1,
                                                         last_epoch=-1)  # Q4: 对应多少步,epoch= 32000/(50000/batch_size),48000,64000
    elif modeltype == 'VGG':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)  # VGG16
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5,
                                                               min_lr=0.000001)  # 动态更新学习率VGG16
        cost = nn.CrossEntropyLoss().to(device)

    Loss_list = []
    Accuracy_list = []
    model.to(device)
    res1 = -1
    res2 = -1
    for epoch in range(epochs):

        model.train()

        training_loss = 0.0
        training_correct = 0
        training_acc = 0.0
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)

        total_train = 0
        for i, (X, y) in tqdm(enumerate(train_loader)):
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

        print('Train acc:{:.4}%'.format(training_acc / total_train))
        if modeltype == 'VGG':
            scheduler.step(training_loss / len(train_loader))
        elif modeltype == 'ResNet':
            scheduler.step()

        model.eval()
        testing_correct = 0
        total = 0
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(test_loader)):
                y = y.long()
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, pred = torch.max(outputs, 1)
                total += y.size(0)
                testing_correct += (pred == y).sum().item()
        t_acc = testing_correct / total


    torch.save(model.state_dict(), model_savepath)


if __name__ == "__main__":
    if args.modeltype == 'ResNet':
        model = ResNet20()
        train(args.datatype, args.epochs,args.modeltype, model,args.model_savepath)
    elif args.modeltype == 'VGG':
        model = VGG('VGG16')
        train(args.datatype, args.epochs,args.modeltype, model,args.model_savepath)