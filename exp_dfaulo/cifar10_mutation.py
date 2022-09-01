import time

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from train_model.models import ResNet20, VGG

device = "cuda"

def retrain(traindatapath, modelsavepath, model, traintype, modeltype):
    trainarr = np.load(traindatapath, allow_pickle=True)
    print(trainarr.shape)
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

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    x_train = torch.from_numpy(np.array([transform_train(x).numpy() for x in deldata[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in deldata[:, 0]]))

    traindataset = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=128, shuffle=True)
    if modeltype == 'ResNet':
        weight_p, bias_p = [], []
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        cost = nn.CrossEntropyLoss().to(device)
        Learning_rate = 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate, momentum=0.9)
        optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay': 1e-4},
                                     {'params': bias_p, 'weight_decay': 0}], lr=Learning_rate,
                                    momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[82, 122, 163], gamma=0.1,
                                                         last_epoch=-1)
    elif modeltype == 'VGG':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5,
                                                               min_lr=0.000001)
        cost = nn.CrossEntropyLoss().to(device)

    Loss_list = []
    Accuracy_list = []
    model.to(device)
    epochs = 1
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

            outputs = model(X)
            loss = cost(outputs, y)
            training_loss += loss.item()
            # print(outputs)
            _, pred = torch.max(outputs, 1)
            total_train += y.size(0)
            num_correct = (pred == y).sum()
            training_acc += num_correct.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print('Train acc:{:.4}%'.format(training_acc / total_train))
        if modeltype == 'VGG':
            scheduler.step(training_loss / len(train_loader))
        elif modeltype == 'ResNet':
            scheduler.step()

    torch.save(model.state_dict(), modelsavepath)

if __name__ == "__main__":
    modelpath = './models/cifar10_alllabel_VGG.pth'
    datapath = './data/CIFA10/CIFA10_PNG/alllabeltraindata_VAE_Kmeans_LOSS_VGG.npy'
    model = VGG('VGG16')
    state_dict = torch.load(modelpath)
    model.load_state_dict(state_dict)
    retrain(datapath, '', model, 'LOSS', 'VGG')
