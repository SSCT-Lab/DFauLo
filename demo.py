import logging

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from pyod.models.vae import VAE
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.cluster import KMeans
from tqdm import tqdm
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from train_model.models import LeNet5

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s -  %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loaddata(datapth):
    logger.info('start loading data.')
    traindata = []
    for label in os.listdir(datapth + 'train'):
        for imgname in os.listdir(datapth + 'train/' + str(label)):
            imgpath = datapth + 'train/' + str(label) + '/' + imgname
            img = cv2.imread(imgpath)
            img = np.array(img, dtype=np.float32)
            traindata.append([label, img, 0])
    traindata = np.array(traindata, dtype=object)
    logger.info('finish data shape:' + str(traindata.shape))
    return traindata


def OAP(dataprg, ratio, model):
    newdata = []
    logger.info('running Outlier Activation and PreLoss:')
    for lb in tqdm(range(10)):
        tmp = []
        for i in range(60000):
            if int(dataprg[i][0]) == lb:
                tmp.append(dataprg[i])
        tmp = np.array(tmp)
        x_train = torch.from_numpy(np.array([x / 255. for x in tmp[:, 1]]))

        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
        ])
        x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))
        y_train = torch.from_numpy(np.array([int(x) for x in tmp[:, 0]]))
        dataset = TensorDataset(x_train, y_train)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=False)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        # Outlier
        x_train_VAE = x_train.reshape(x_train.shape[0], 28 * 28)
        clf_vae = VAE(epochs=5, verbose=0)
        clf_vae.fit(x_train_VAE)
        vae_score = clf_vae.decision_scores_
        vae_score = np.array(vae_score)

        # Activation and PreLoss
        model.eval()
        model.to(device)
        Act = []
        PreLoss = []
        with torch.no_grad():
            for batch, (X, y) in enumerate(data_loader):
                y = y.long()
                X, y = X.to(device), y.to(device)
                out = model.c1(X)
                out = model.Sigmoid(out)
                out = model.s2(out)
                out = model.c3(out)
                out = model.Sigmoid(out)
                out = model.s4(out)
                out = model.c5(out)
                out = model.flatten(out)
                Act_out = out.cpu().numpy()
                out = model.f6(out)
                out = model.f7(out)
                Loss = loss_fn(out, y).cpu().numpy()
                for i in range(Act_out.shape[0]):
                    Act.append(Act_out[i])
                    PreLoss.append(Loss[i])
        Act = np.array(Act)
        PreLoss = np.array(PreLoss)

        clf_kmeans = KMeans(n_clusters=2)
        clf_kmeans.fit(Act)
        km_label = clf_kmeans.labels_

        label1num = 0
        label0num = 0
        for i in range(km_label.shape[0]):
            if km_label[i] == 1:
                label1num += 1
            elif km_label[i] == 0:
                label0num += 1

        seldata = -1
        if label1num <= label0num:
            seldata = 0
        else:
            seldata = 1

        res = []
        for i in range(tmp.shape[0]):
            if km_label[i] == seldata:
                res.append([tmp[i][0], tmp[i][1], tmp[i][2], vae_score[i], 0, PreLoss[i]])
            else:
                res.append([tmp[i][0], tmp[i][1], tmp[i][2], vae_score[i], 1, PreLoss[i]])

        res = np.array(res, dtype=object)
        res = res[res[:, 3].argsort()[::-1]]  # vae score sort
        for i in range(res.shape[0]):
            if i <= int(res.shape[0] * ratio):
                res[i, 3] = 1
            else:
                res[i, 3] = 0
        res = res[res[:, 5].argsort()[::-1]]  # loss sort
        for i in range(res.shape[0]):
            if i <= int(res.shape[0] * ratio):
                res[i, 5] = 1
            else:
                res[i, 5] = 0
            newdata.append(res[i])
    logger.info('finish OAP.')
    newdata = np.array(newdata, dtype=object)
    return newdata


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


def mutate_LeNet5(model, trainarr, newmodelsavepath, traintype):
    lr = 0.001
    epoch = 10

    delind = -1
    if traintype == 'Outlier':
        delind = 3
    elif traintype == 'Activation':
        delind = 4
    elif traintype == 'PreLoss':
        delind = 5

    deldata = []
    for i in range(trainarr.shape[0]):
        if int(trainarr[i][delind]) == 0:
            deldata.append([trainarr[i][0], trainarr[i][1]])
    data = np.array(deldata)
    logger.info('mutation data shape:' + str(data.shape))

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
    logger.info('start mutate ' + traintype + ' model:')
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
    torch.save(model.state_dict(), newmodelsavepath)


if __name__ == "__main__":
    # data = loaddata('H:/ASEprj/Code/data/MNIST/MNIST_PNG/')
    orgdata = np.load('H:\\ASEprj\\Code\\data\\MNIST\MNIST_PNG\\alllabeltraindata.npy', allow_pickle=True)

    model = LeNet5()
    state_dict = torch.load('./demodata/mnist_RandomLabelNoise_LeNet5.pth')
    model.load_state_dict(state_dict)

    OPAdata = OAP(dataprg=orgdata,
                  ratio=0.05,
                  model=model)

    for mutype in ['Outlier', 'Activation', 'PreLoss']:
        mutate_LeNet5(model=model,
                      trainarr=OPAdata,
                      newmodelsavepath='./demodata/' + mutype + '.pth',
                      traintype=mutype)
