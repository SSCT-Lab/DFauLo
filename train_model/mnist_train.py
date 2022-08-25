import argparse

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

parser = argparse.ArgumentParser()
from models import LeNet5, LeNet1

parser.add_argument('--modeltype', type=str, default='LeNet1', help='LeNet1 or LeNet5.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--epoch', type=int, default=85, help='epochs')
parser.add_argument('--traindatapth', type=str, default='./data/MNIST/MNIST_PNG/orgtraindata.npy',
                    help='train datapath.')
parser.add_argument('--testdatapath', type=str, default='./data/MNIST/MNIST_PNG/orgtestdata.npy', help='test datapath.')
parser.add_argument('--modelsave_path', type=str, default='./models/mnist_LeNet1.pth', help='model save path.')

args = parser.parse_args()
device = 'cuda'


def val(dataloader, model, loss_fn):

    model.eval()
    loss, current, n = 0.0, 0.0, 0

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
        return current / n


def acc_in_orgtestset(model, loss_fn):
    print("model in org datasrt:\n")
    datapath = './data/MNIST/MNIST_PNG/orgtestdata.npy'
    testarr = np.load(datapath, allow_pickle=True)
    x_test = torch.from_numpy(np.array([x / 255. for x in testarr[:, 1]]))
    y_test = torch.from_numpy(np.array([int(x) for x in testarr[:, 0]]))

    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
    ])
    x_test = data_transform(x_test.reshape(x_test.shape[0], 3, 28, 28))

    testdataset = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=16, shuffle=True)

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


        return current / n


def train_data(model, lr, epoch, traindatapth, testdatapath, modelsave_path):
    trainarr = np.load(traindatapth, allow_pickle=True)
    testarr = np.load(testdatapath, allow_pickle=True)

    x_train = torch.from_numpy(np.array([x / 255. for x in trainarr[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in trainarr[:, 0]]))

    x_test = torch.from_numpy(np.array([x / 255. for x in testarr[:, 1]]))
    y_test = torch.from_numpy(np.array([int(x) for x in testarr[:, 0]]))

    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
    ])
    x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))
    x_test = data_transform(x_test.reshape(x_test.shape[0], 3, 28, 28))

    traindataset = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=16, shuffle=True)

    testdataset = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=16, shuffle=True)

    min_acc = 0
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.to(device)
    res1 = -1
    res2 = -1
    for t in range(epoch):
        loss, current, n = 0.0, 0.0, 0
        model.train()
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

        a = val(test_loader, model, loss_fn)
        if a > res1:
            res1 = a
            res2 = current / n
            torch.save(model.state_dict(), modelsave_path)
        print('now max train:' + ' ' + str(res2) + ' test: ' + str(res1))
        print(f"epoch{t + 1} loss{a}\n-------------------")
        if t == epoch - 1:
            torch.save(model.state_dict(), modelsave_path)
            acc_in_orgtestset(model, loss_fn)
    print("DONE")


if __name__ == "__main__":
    if args.modeltype == 'LeNet1':
        model = LeNet1()
        train_data(model, args.lr, args.epoch, args.traindatapth, args.testdatapath, args.modelsave_path)
    elif args.modeltype == 'LeNet5':
        model = LeNet5()
        train_data(model, args.lr, args.epoch, args.traindatapth, args.testdatapath, args.modelsave_path)
