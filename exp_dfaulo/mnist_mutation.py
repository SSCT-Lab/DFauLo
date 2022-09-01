
import numpy as np
import torch

from torch import nn
from torch.utils.data import TensorDataset
from torchvision import transforms

from train_model.models import LeNet1, LeNet5
def getpremodel_LeNet1(modelpath):
    premodel = LeNet1()
    state_dict = torch.load(modelpath)
    premodel.load_state_dict(state_dict)
    for param in premodel.parameters():
        param.requires_grad = False
    premodel.fc = nn.Sequential(
        nn.Linear(12 * 4 * 4, 10)
    )
    return premodel


def getpremodel_LeNet5(modelpath):
    premodel = LeNet5()
    state_dict = torch.load(modelpath)
    premodel.load_state_dict(state_dict)
    for param in premodel.parameters():
        param.requires_grad = False
    premodel.f6 = nn.Sequential(
        nn.Linear(120, 84)
    )
    premodel.f7 = nn.Sequential(
        nn.Linear(84, 10)
    )
    return premodel

def mutate_LeNet1(datapath, premodelpath, newmodelsavepath, weighttype):
    lr = 0.001
    device = 'cuda'
    epoch = 10
    if weighttype == 'randomweight':
        model = getpremodel_LeNet1(premodelpath)
    elif weighttype == 'direct':
        model = LeNet1()
        state_dict = torch.load(premodelpath)
        model.load_state_dict(state_dict)
    data = np.load(datapath, allow_pickle=True)
    x_train = torch.from_numpy(np.array([x / 255. for x in data[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in data[:, 0]]))
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
    ])
    x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))
    traindataset = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=16, shuffle=True)
    min_acc = 0
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.to(device)
    for t in range(epoch):
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
        print('train_loss' + str(loss / n))
        print('train_acc' + str(current / n))
        print(f"epoch{t + 1} -------------------")
        if t == epoch - 1:
            torch.save(model.state_dict(), newmodelsavepath)
    print("DONE")

def mutate_LeNet5(datapath, premodelpath, newmodelsavepath, weighttype):
    lr = 0.001
    device = 'cuda'
    epoch = 10

    if weighttype == 'randomweight':
        model = getpremodel_LeNet5(premodelpath)
    elif weighttype == 'direct':
        model = LeNet5()
        state_dict = torch.load(premodelpath)
        model.load_state_dict(state_dict)

    data = np.load(datapath, allow_pickle=True)

    # data = []
    # for i in range(predata.shape[0]):
    #     if int(predata[i][2]) == 0:
    #         data.append(predata[i])
    # data = np.array(data)

    x_train = torch.from_numpy(np.array([x / 255. for x in data[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in data[:, 0]]))
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
    ])
    x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))
    traindataset = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=16, shuffle=True)
    min_acc = 0
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.to(device)
    for t in range(epoch):
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
        print('train_loss' + str(loss / n))
        print('train_acc' + str(current / n))
        print(f"epoch{t + 1} -------------------")
        if t == epoch - 1:
            torch.save(model.state_dict(), newmodelsavepath)
    print("DONE")

if __name__ == "__main__":
    datapath = './data/MNIST/MNIST_PNG/alllabeltraindata_Outlier.npy'
    premodelpath = './models/mnist_RandomLabelNoise_LeNet5.pth'
    newmodelsavepath = './models/mnist_alllabel_LeNet5_Outlier.pth'
    mutate_LeNet5(datapath, premodelpath, newmodelsavepath, 'direct')