import copy
import logging
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from pyod.models.vae import VAE
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
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
        Sfmout = []
        softmax_func = nn.Softmax(dim=1)
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
                Loss = loss_fn(softmax_func(out), y).cpu().numpy()
                SFM = softmax_func(out).cpu().numpy()
                for i in range(Act_out.shape[0]):
                    Act.append(Act_out[i])
                    PreLoss.append(Loss[i])
                    Sfmout.append(SFM[i])
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
                res.append([tmp[i][0], tmp[i][1], tmp[i][2], vae_score[i], 0, PreLoss[i],
                            Sfmout[i], PreLoss[i]])
            else:
                res.append([tmp[i][0], tmp[i][1], tmp[i][2], vae_score[i], 1, PreLoss[i],
                            Sfmout[i], PreLoss[i]])

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


def mutate_LeNet5(model, trainarr, traintype):
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
    data = np.array(deldata, dtype=object)
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

    # getfeature
    model.eval()
    SfmOut = []
    PreLoss = []
    x_train = torch.from_numpy(np.array([x / 255. for x in trainarr[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in trainarr[:, 0]]))
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
    ])
    x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))
    traindataset = TensorDataset(x_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=64, shuffle=False)
    softmax_func = nn.Softmax(dim=1)
    loss_fn2 = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            y = y.long()
            X, y = X.to(device), y.to(device)
            out = model(X)
            Loss = loss_fn2(softmax_func(out), y).cpu().numpy()
            SFM = softmax_func(out).cpu().numpy()
            for i in range(Loss.shape[0]):
                PreLoss.append(Loss[i])
                SfmOut.append(SFM[i])
    SfmOut = np.array(SfmOut)
    PreLoss = np.array(PreLoss)
    return SfmOut, PreLoss, model


def getmodelout(model, X, Y):
    model.eval()
    sfout = []
    softmax_func = nn.Softmax(dim=1)
    X = X.astype('float32')
    X = torch.from_numpy(X)
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        X = X.to(device)
        out = model(X)
        y = []
        y.append(Y)
        y = np.array(y)
        y = torch.from_numpy(y)
        y = y.long()
        y = y.to(device)
        soft_output = softmax_func(out)
        loss = loss_fn(soft_output, y).cpu().numpy().item()
        sfout = soft_output.cpu().numpy()[0]

    return sfout, loss


def getrandomdata(lb, Outlier_model, Activation_model, PreLoss_model, org_model):
    X = np.zeros((1, 1, 28, 28))
    # print(X.shape)
    for i in range(28):
        for j in range(28):
            X[0, 0, i, j] = random.random()
    Y = lb

    orgsfm, orgloss = getmodelout(org_model, X, Y)
    S1sfm, S1loss = getmodelout(Outlier_model, X, Y)

    tmp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tmp[int(Y)] = 1

    S2sfm, S2loss = getmodelout(Activation_model, X, Y)
    S3sfm, S3loss = getmodelout(PreLoss_model, X, Y)

    fea = []
    for i in range(10):
        fea.append(orgsfm[i])
    for i in range(10):
        fea.append(tmp[i])
    for i in range(10):
        fea.append(S1sfm[i])
    for i in range(10):
        fea.append(S2sfm[i])
    for i in range(10):
        fea.append(S3sfm[i])
    fea.append(1)
    fea.append(1)
    fea.append(1)
    fea.append(orgloss)
    fea.append(S1loss)
    fea.append(S2loss)
    fea.append(S3loss)

    fea = np.array(fea)
    return fea


def Susp_initialization(feaVec, Outlier_model, Activation_model, PreLoss_model, org_model):
    logger.info('start Offline')
    isdt = np.array([int(x) for x in feaVec[:, -1]])
    feaVecsimple = feaVec[:, 0:-3]
    NUM = 100
    NUM2 = 10

    cnt = 0
    newfea = []
    tmpfea = copy.deepcopy(feaVecsimple)
    ind = [x for x in range(tmpfea.shape[0])]
    random.shuffle(ind)
    tmpfea = tmpfea[ind]

    for i in range(tmpfea.shape[0]):
        if cnt < NUM:
            newfea.append(tmpfea[i])
            cnt += 1
    for i in range(NUM2):
        newfea.append(getrandomdata(i, Outlier_model, Activation_model, PreLoss_model, org_model))

    newfea = np.array(newfea)

    Y = []
    for i in range(NUM):
        Y.append(0)
    for i in range(NUM2):
        Y.append(1)
    Y = np.array(Y)

    lg = LogisticRegression(C=1.0)
    lg.fit(newfea, Y)

    LRres = lg.predict_proba(feaVecsimple)  ####@@@@
    LRres = LRres[:, 1]

    newdata = []
    for i in range(isdt.shape[0]):
        newdata.append([feaVec[i][-3], feaVec[i][-2], feaVec[i][-1], float(LRres[i])])

    newdata = np.array(newdata, dtype=object)
    feaVec = feaVec[newdata[:, 3].argsort()[::-1]]

    newdata = newdata[newdata[:, 3].argsort()[::-1]]

    return feaVec, newdata


def visualization(offline_data):
    for i in range(50):
        img = Image.fromarray(offline_data[i][1].reshape(28, 28, 3).astype('uint8'))
        img = ImageOps.invert(img)
        img.save('./demodata/DFaLo_offline_result/' +
                 str(i) + '_label_' + str(offline_data[i][0]) + '.png')


def DfauLo(fea_labeled, label, fea_left):

    lg = LogisticRegression(C=1.0)
    lg.fit(fea_labeled, label)

    LRres = lg.predict_proba(fea_left)  ####@@@@
    LRres = LRres[:, 1]

    fea_left = fea_left[LRres.argsort()[::-1]]

    return fea_left


if __name__ == "__main__":
    orgdata = loaddata('./demodata/MNIST/')
    # orgdata = np.load('H:\\ASEprj\\Code\\data\\MNIST\MNIST_PNG\\alllabeltraindata.npy', allow_pickle=True)

    model = LeNet5()
    state_dict = torch.load('./demodata/mnist_original_LeNet5.pth')
    model.load_state_dict(state_dict)

    OPAdata = OAP(dataprg=orgdata,
                  ratio=0.05,
                  model=model)
    del orgdata

    SfmOut_Outlier, PreLoss_Outlier, Outlier_model = mutate_LeNet5(model=model,
                                                                   trainarr=OPAdata,
                                                                   traintype='Outlier')
    SfmOut_Activation, PreLoss_Activation, Activation_model = mutate_LeNet5(model=model,
                                                                            trainarr=OPAdata,
                                                                            traintype='Activation')
    SfmOut_PreLoss, PreLoss_PreLoss, PreLoss_model = mutate_LeNet5(model=model,
                                                                   trainarr=OPAdata,
                                                                   traintype='Activation')

    logger.info('feature summary')
    Feature = []
    # feature summary
    for i in tqdm(range(OPAdata.shape[0])):
        fea = []
        # original model softmax
        for j in range(10):
            fea.append(OPAdata[i, 6][j])
        # ground truth
        tmp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        tmp[int(OPAdata[i, 0])] = 1
        for j in range(10):
            fea.append(tmp[j])

        # Outlier mutation model softmax
        for j in range(10):
            fea.append(SfmOut_Outlier[i][j])

        # Activation mutation model softmax
        for j in range(10):
            fea.append(SfmOut_Activation[i][j])

        # PreLoss mutation model softmax
        for j in range(10):
            fea.append(SfmOut_PreLoss[i][j])

        # is Outlier in top ratio
        fea.append(OPAdata[i, 3])
        # is Activation in top ratio
        fea.append(OPAdata[i, 4])
        # is PreLoss in top ratio
        fea.append(OPAdata[i, 5])

        # original mutation model Loss
        fea.append(OPAdata[i, 7])
        # Oulier mutation model Loss
        fea.append(PreLoss_Outlier[i])
        # Activation mutation model Loss
        fea.append(PreLoss_Activation[i])
        # PreLoss mutation model Loss
        fea.append(PreLoss_PreLoss[i])

        # label
        fea.append(OPAdata[i, 0])
        # data
        fea.append(OPAdata[i, 1])
        # is a dirty data
        fea.append(OPAdata[i, 2])

        Feature.append(fea)
    Feature = np.array(Feature, dtype=object)

    Si_fea, Si_data = Susp_initialization(feaVec=Feature,
                                        Outlier_model=Outlier_model,
                                        Activation_model=Activation_model,
                                        PreLoss_model=PreLoss_model,
                                        org_model=model)

    np.save('./demodata/Si_data.npy', Si_data)

    # Si_data = np.load('./demodata/Si_data.npy', allow_pickle=True)
    visualization(Si_data)

    '''
    DfauLo part need, human labeling
    '''
    for round in range(10):

        '''
        label the first n data to obtain the labeled data corresponding feature(fea_labeled), 
        corresponding labels(label) and remaining data corresponding feature(fea_left)
        '''

        fea_labeled = np.load('./demodata/fea_labeled'+str(round)+'.npy', allow_pickle=True)
        label = np.load('./demodata/label'+str(round)+'.npy', allow_pickle=True)
        fea_left = np.load('./demodata/fea_left'+str(round)+'.npy', allow_pickle=True)

        DfauLo(fea_labeled, label, fea_left)


