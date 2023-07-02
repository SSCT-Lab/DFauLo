import copy
import random
import time

import numpy as np
import torch
import xlrd
import xlwt
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from numpy import mat
from torch import nn
from pyod.models.vae import VAE
from pyod.models.iforest import IForest
from torchvision import transforms
from sklearn.cluster import KMeans
from _exp_.train_model.models import LeNet5, LeNet1
from sklearn.linear_model import LogisticRegression


def PoBL(ranklist, ratio):
    n = 60000
    m = 0
    for i in range(n):
        if ranklist[i][0] == 1:
            m += 1
    cnt = 0
    for i in range(int(n * ratio)):
        if ranklist[i][0] == 1:
            cnt += 1
    # print('PoBL score: ', cnt / m)
    return cnt / m


def RAUC(ranklist, bestlist):
    rat = [x for x in range(101)]
    y = []
    yb = []
    for i in rat:
        y.append(PoBL(ranklist, i / 100.) * 100)
        yb.append(PoBL(bestlist, i / 100.) * 100)
    colorlist = ['violet', 'green', 'red', 'hotpink', 'mediumblue', 'orange', 'yellow', 'yellowgreen', 'peachpuff']
    # plt.plot(rat, y, color=colorlist[PCNT], label=str(PCNT + 1))
    # if PCNT==0:
    # plt.plot(rat, yb, color='blue', label='best')
    # plt.legend()
    # if PCNT==PALL-1:
    # plt.show()
    # print("RAUC score: ", np.trapz(y, rat) / np.trapz(yb, rat))
    return np.trapz(y, rat) / np.trapz(yb, rat)


def bestAUC(ranklist):
    bestlist = []
    for i in range(60000):
        if ranklist[i][0] == 1:
            bestlist.append(ranklist[i])
    for i in range(60000):
        if ranklist[i][0] == 0:
            bestlist.append(ranklist[i])
    bestlist = np.array(bestlist)
    return bestlist


def EXAM_F(ranklist):
    n = 60000
    pos = -1
    for i in range(n):
        if ranklist[i][0] == 1:
            pos = i
            break
    return (pos + 1) / n
    # print('EXAM_F score: ', (pos + 1) / n)


def EXAM_L(ranklist):
    n = 60000
    pos = -1
    for i in range(n - 1, -1, -1):
        if ranklist[i][0] == 1:
            pos = i
            break
    return (pos + 1) / n
    # print('EXAM_L score: ', (pos + 1) / n)


def EXAM_AVG(ranklist):
    n = 60000
    m = 0
    tf = 0
    for i in range(n):
        if ranklist[i][0] == 1:
            tf += i
            m += 1
    return tf / (n * m)
    # print('EXAM_AVG score: ', tf / (n * m))


def APFD(ranklist):
    n = 60000
    m = 0
    tf = 0
    for i in range(n):
        if ranklist[i][0] == 1:
            tf += i
            m += 1

    # print('APFD score: ', 1 - (tf / (n * m)) + (1 / (2 * n)))
    return 1 - (tf / (n * m)) + (1 / (2 * n))


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


def getOUTPUT(model, datapath):
    predata = np.load(datapath, allow_pickle=True)
    x_train = torch.from_numpy(np.array([x / 255. for x in predata[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in predata[:, 0]]))
    is_dirty = torch.from_numpy(np.array([int(x) for x in predata[:, 2]]))
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
    ])
    x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))
    model.eval()
    sfout = []
    softmax_func = nn.Softmax(dim=1)

    with torch.no_grad():
        for i in range(x_train.shape[0]):
            out = model(x_train[i].reshape(1, 1, 28, 28))
            soft_output = softmax_func(out)
            sfout.append(soft_output.numpy()[0])

    sfout = np.array(sfout)
    return sfout


def getGT(datapath):
    predata = np.load(datapath, allow_pickle=True)
    y_train = torch.from_numpy(np.array([int(x) for x in predata[:, 0]]))

    GTsf = []
    for i in range(y_train.shape[0]):
        tmp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        tmp[int(y_train[i])] = 1
        GTsf.append(tmp)

    GTsf = np.array(GTsf)

    return GTsf


def ISO(datapath):
    dataprg = np.load(datapath, allow_pickle=True)
    newdata = []
    for lb in range(10):
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
        x_train = x_train.reshape(x_train.shape[0], 28 * 28)
        clf = IForest()
        clf.fit(x_train)
        y_score = clf.decision_scores_
        y_label = clf.labels_
        y_label = np.array(y_label)
        y_score = np.array(y_score)

        res = []
        for i in range(tmp.shape[0]):
            res.append([0, i, y_score[i]])
        res = np.array(res, dtype=object)

        res = res[res[:, 2].argsort()[::-1]]

        for i in range(int(res.shape[0] * 0.05)):
            res[i][0] = 1

        res = res[res[:, 1].argsort()]

        for i in range(res.shape[0]):
            newdata.append(res[i])

    newdata = np.array(newdata)
    print(newdata.shape)

    return newdata


def clusterLeNet1(model, datapath):
    dataprg = np.load(datapath, allow_pickle=True)
    newdata = []
    for lb in range(10):
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

        res = []
        with torch.no_grad():
            for i in range(x_train.shape[0]):
                out = model.c1(x_train[i].reshape(1, 1, 28, 28))
                out = model.TANH(out)
                out = model.s2(out)
                out = model.c3(out)
                out = model.TANH(out)
                out = model.s4(out)
                out = out.view(out.size(0), -1)
                out = out.numpy()
                out = out.reshape(out.shape[1])
                res.append(out)
        res = np.array(res)
        print(res.shape)
        clf = KMeans(n_clusters=2)
        clf.fit(res)
        y_label = clf.labels_
        cnt = 0
        sum = 0
        for i in range(y_label.shape[0]):
            if y_label[i] == 1:
                cnt += 1
            elif y_label[i] == 0:
                sum += 1
        print(cnt, sum)

        seldata = -1
        if cnt <= sum:
            seldata = 1
        else:
            seldata = 0

        selnum = 0
        for i in range(tmp.shape[0]):
            if y_label[i] == seldata:
                newdata.append(1)
                selnum += 1
            else:
                newdata.append(0)
        print(selnum)
    newdata = np.array(newdata, dtype=object)
    return newdata


def clusterLeNet5(model, datapath):
    dataprg = np.load(datapath, allow_pickle=True)
    newdata = []
    for lb in range(10):
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

        res = []
        with torch.no_grad():
            for i in range(x_train.shape[0]):
                out = model.c1(x_train[i].reshape(1, 1, 28, 28))
                out = model.Sigmoid(out)
                out = model.s2(out)
                out = model.c3(out)
                out = model.Sigmoid(out)
                out = model.s4(out)
                out = model.c5(out)
                out = model.flatten(out)
                out = out.numpy()
                out = out.reshape(out.shape[1])
                res.append(out)
        res = np.array(res)
        print(res.shape)
        clf = KMeans(n_clusters=2)
        clf.fit(res)
        y_label = clf.labels_
        cnt = 0
        sum = 0
        for i in range(y_label.shape[0]):
            if y_label[i] == 1:
                cnt += 1
            elif y_label[i] == 0:
                sum += 1
        print(cnt, sum)

        seldata = -1
        if cnt <= sum:
            seldata = 1
        else:
            seldata = 0

        selnum = 0
        for i in range(tmp.shape[0]):
            if y_label[i] == seldata:
                newdata.append(1)
                selnum += 1
            else:
                newdata.append(0)
        print(selnum)
    newdata = np.array(newdata, dtype=object)
    return newdata


def Deletedata_S1(model, predatapath):
    predata = np.load(predatapath, allow_pickle=True)
    x_train = torch.from_numpy(np.array([x / 255. for x in predata[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in predata[:, 0]]))
    is_dirty = torch.from_numpy(np.array([int(x) for x in predata[:, 2]]))
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
    ])
    x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
    ranklist = [[], [], [], [], [], [], [], [], [], []]
    with torch.no_grad():
        for i in range(60000):
            out = model(x_train[i].reshape(1, 1, 28, 28))
            y = []
            y.append(y_train[i])
            y = np.array(y)
            y = torch.from_numpy(y)
            y = y.long()
            tt, pred = torch.max(out, axis=1)
            cur_loss = loss_fn(out, y)
            ranklist[int(predata[i][0])].append([predata[i][0], predata[i][1], predata[i][2], float(cur_loss)])
    newdata = []
    for i in range(10):
        tmp = []
        for j in range(len(ranklist[i])):
            tmp.append([0, ranklist[i][j][3], j])
        tmp = np.array(tmp, dtype=object)
        tmp = tmp[tmp[:, 1].argsort()[::-1]]

        for k in range(int(tmp.shape[0] * 0.05)):
            tmp[k][0] = 1

        tmp = tmp[tmp[:, 2].argsort()]

        for j in range(tmp.shape[0]):
            newdata.append(tmp[j])

    newdata = np.array(newdata)
    return newdata


def Cross_Entropy_Loss(X, GT):
    softmax_func = nn.Softmax(dim=1)
    soft_output = softmax_func(X)
    GTs = softmax_func(GT)
    log_output = torch.log(soft_output)
    sum = 0
    for i in range(10):
        sum += GTs[0][i] * log_output[0][i]
    return -sum


def susp(orgmodel, S1model, S2model, S3model, datapath):
    predata = np.load(datapath, allow_pickle=True)
    x_train = torch.from_numpy(np.array([x / 255. for x in predata[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in predata[:, 0]]))
    is_dirty = torch.from_numpy(np.array([int(x) for x in predata[:, 2]]))
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
    ])
    x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失

    susplist = []

    with torch.no_grad():
        for i in range(x_train.shape[0]):
            y = []
            y.append(y_train[i])
            y = np.array(y)
            y = torch.from_numpy(y)
            y = y.long()

            softmax_func = nn.Softmax(dim=1)
            org_out = softmax_func(orgmodel(x_train[i].reshape(1, 1, 28, 28)))
            S1_out = softmax_func(S1model(x_train[i].reshape(1, 1, 28, 28)))
            S2_out = softmax_func(S2model(x_train[i].reshape(1, 1, 28, 28)))
            S3_out = softmax_func(S3model(x_train[i].reshape(1, 1, 28, 28)))

            org_loss = loss_fn(org_out, y)
            S1_loss = loss_fn(S1_out, y)
            S2_loss = loss_fn(S2_out, y)
            S3_loss = loss_fn(S3_out, y)

            susplist.append(S1_loss - org_loss + S2_loss - org_loss + S3_loss - org_loss)
    susplist = np.array(susplist)
    return susplist


def Detial_susp(orgmodel, S1model, S2model, S3model, datapath):
    predata = np.load(datapath, allow_pickle=True)
    x_train = torch.from_numpy(np.array([x / 255. for x in predata[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in predata[:, 0]]))
    is_dirty = torch.from_numpy(np.array([int(x) for x in predata[:, 2]]))
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
    ])
    x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失

    orglist = []
    S1list = []
    S2list = []
    S3list = []

    with torch.no_grad():
        for i in range(x_train.shape[0]):
            y = []
            y.append(y_train[i])
            y = np.array(y)
            y = torch.from_numpy(y)
            y = y.long()

            softmax_func = nn.Softmax(dim=1)
            org_out = softmax_func(orgmodel(x_train[i].reshape(1, 1, 28, 28)))
            S1_out = softmax_func(S1model(x_train[i].reshape(1, 1, 28, 28)))
            S2_out = softmax_func(S2model(x_train[i].reshape(1, 1, 28, 28)))
            S3_out = softmax_func(S3model(x_train[i].reshape(1, 1, 28, 28)))

            org_loss = loss_fn(org_out, y)
            S1_loss = loss_fn(S1_out, y)
            S2_loss = loss_fn(S2_out, y)
            S3_loss = loss_fn(S3_out, y)

            orglist.append(org_loss)
            S1list.append(S1_loss)
            S2list.append(S2_loss)
            S3list.append(S3_loss)

    return orglist, S1list, S2list, S3list


def get_susp_rank_res(suslist, is_dirty):
    rank = []
    for i in range(is_dirty.shape[0]):
        rank.append([int(is_dirty[i]), float(suslist[i])])
    rank = np.array(rank)
    rank = rank[rank[:, 1].argsort()[::-1]]
    print(rank)
    print("rank shape: ", rank.shape)
    RAUC(rank, bestAUC(rank))
    EXAM_F(rank)
    EXAM_L(rank)
    EXAM_AVG(rank)
    PoBL(rank, 0.1)
    APFD(rank)


def get_featureV(orgsfm, GTsfm, S1sfm, S2sfm, S3sfm, ISS1, ISS2, ISS3, suslist, orgloss, S1loss, S2loss, S3loss,
                 is_dirty, datatype, modeltype, predata, dataratio):
    feaV = []

    for i in range(is_dirty.shape[0]):
        tmp = []
        for j in range(10):
            tmp.append(orgsfm[i][j])
        for j in range(10):
            tmp.append(GTsfm[i][j])
        for j in range(10):
            tmp.append(S1sfm[i][j])
        for j in range(10):
            tmp.append(S2sfm[i][j])
        for j in range(10):
            tmp.append(S3sfm[i][j])
        tmp.append(ISS1[i][0])
        tmp.append(ISS2[i])
        tmp.append(ISS3[i][0])
        tmp.append(orgloss[i])
        tmp.append(S1loss[i])
        tmp.append(S2loss[i])
        tmp.append(S3loss[i])
        tmp.append(suslist[i])
        tmp.append(int(is_dirty[i]))
        # tmp.append(predata[i][1])  ##CLEAN LAB
        # tmp.append(predata[i][0])  ##
        feaV.append(tmp)

    feaV = np.array(feaV)

    np.save('./data/MNIST/MNIST_LR_feature/' + dataratio + '_' + datatype + '_feature_' + modeltype + '.npy', feaV)


all_key = []

from pynput.keyboard import Listener

lb_LR = []


def on_press(key):
    all_key.append(str(key))

    if "'1'" in all_key:
        lb_LR.append(1)
    elif "'0'" in all_key:
        lb_LR.append(0)
    all_key.clear()
    return False


def start_listen():
    with Listener(on_press=on_press, on_release=None) as listener:
        listener.join()


def cls1(feaVec):
    num = 100
    feaVec[:, -2] = feaVec[:, -2] + 2 * feaVec[:, -6]
    feaVec = feaVec[feaVec[:, -2].argsort()[::-1]]

    X = np.vstack((feaVec[0:num, 0:-1], feaVec[-num:60000, 0:-1]))  ####@@@@

    Y = np.concatenate((feaVec[0:num, -1], feaVec[-num:60000, -1]))

    if 1 not in Y:
        print('no negative feature')
        indlist = [x for x in range(100, 60000 - 100)]
        random.shuffle(indlist)
        cnt = 0
        while 1 not in Y:
            cnt += 1
            for i in range(cnt * 50, cnt * 50 + 50):

                tmp = []
                for z in range(feaVec[indlist[i]].shape[0] - 1):
                    tmp.append(feaVec[indlist[i]][z])
                print(X.shape)
                X = np.row_stack((X, tmp))
                print(X.shape)
                Y = np.append(Y, feaVec[indlist[i], -1])
        file = open('run.txt', 'a')
        s = 'optimize ' + str(cnt) + ' times,' + 'dirty:   ' + str(Y.sum()) + '\n'
        file.write(s)
        file.close()
        print('optimize ', cnt, ' times')

    print(X.shape, Y.shape)

    lg = LogisticRegression(C=1.0)
    lg.fit(X, Y)

    LRres = lg.predict_proba(feaVec[0:60000, 0:-1])  ####@@@@
    LRres = LRres[:, 1]

    rank = []
    for i in range(feaVec.shape[0]):
        rank.append([int(feaVec[i][-1]), float(LRres[i])])

    rank = np.array(rank)
    rank = rank[rank[:, 1].argsort()[::-1]]
    print(rank)
    print("rank shape: ", rank.shape)
    RAUC(rank, bestAUC(rank))
    EXAM_F(rank)
    EXAM_L(rank)
    EXAM_AVG(rank)
    PoBL(rank, 0.1)
    APFD(rank)


def getmodelout(model, X, Y):
    model.eval()
    sfout = []
    softmax_func = nn.Softmax(dim=1)
    X = X.astype('float32')
    X = torch.from_numpy(X)

    with torch.no_grad():
        out = model(X)
        soft_output = softmax_func(out)
        sfout.append(soft_output.numpy()[0])

    sfout = np.array(sfout)
    return sfout


def getransusp(orgmodel, S1model, S2model, S3model, X, Y):
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
    X = X.astype('float32')
    X = torch.from_numpy(X)
    orglist = []
    S1list = []
    S2list = []
    S3list = []

    with torch.no_grad():
        y = []
        y.append(Y)
        y = np.array(y)
        y = torch.from_numpy(y)
        y = y.long()

        softmax_func = nn.Softmax(dim=1)
        org_out = softmax_func(orgmodel(X))
        S1_out = softmax_func(S1model(X))
        S2_out = softmax_func(S2model(X))
        S3_out = softmax_func(S3model(X))

        org_loss = loss_fn(org_out, y)
        S1_loss = loss_fn(S1_out, y)
        S2_loss = loss_fn(S2_out, y)
        S3_loss = loss_fn(S3_out, y)

        orglist.append(org_loss.item())
        S1list.append(S1_loss.item())
        S2list.append(S2_loss.item())
        S3list.append(S3_loss.item())

    return orglist, S1list, S2list, S3list


def getrandomdata(modeltype, datatype, lb, ablation):
    X = np.zeros((1, 1, 28, 28))
    # print(X.shape)
    for i in range(28):
        for j in range(28):
            X[0, 0, i, j] = random.random()
    Y = lb

    if modeltype == 'LeNet5':
        orgmodel = LeNet5()
        S1model = getpremodel_LeNet5('./models/mnist_' + datatype + '_LeNet5.pth')
        S2model = getpremodel_LeNet5('./models/mnist_' + datatype + '_LeNet5.pth')
        S3model = getpremodel_LeNet5('./models/mnist_' + datatype + '_LeNet5.pth')
    elif modeltype == 'LeNet1':
        orgmodel = LeNet1()
        S1model = getpremodel_LeNet1('./models/mnist_' + datatype + '_LeNet1.pth')
        S2model = getpremodel_LeNet1('./models/mnist_' + datatype + '_LeNet1.pth')
        S3model = getpremodel_LeNet1('./models/mnist_' + datatype + '_LeNet1.pth')

    org_state_dict = torch.load('./models/mnist_' + datatype + '_' + modeltype + '.pth')
    orgmodel.load_state_dict(org_state_dict)

    S1_state_dict = torch.load('./retrainmodel/mnist_' + datatype + '_' + modeltype + '_retrain_VAE.pth')
    S1model.load_state_dict(S1_state_dict)

    S2_state_dict = torch.load('./retrainmodel/mnist_' + datatype + '_' + modeltype + '_retrain_Kmeans.pth')
    S2model.load_state_dict(S2_state_dict)

    S3_state_dict = torch.load('./retrainmodel/mnist_' + datatype + '_' + modeltype + '_retrain.pth')
    S3model.load_state_dict(S3_state_dict)

    orgsfm = getmodelout(orgmodel, X, Y)
    S1sfm = getmodelout(S1model, X, Y)
    GTsf = []

    tmp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tmp[int(Y)] = 1
    GTsf.append(tmp)
    GTsf = np.array(GTsf)
    S2sfm = getmodelout(S2model, X, Y)
    S3sfm = getmodelout(S3model, X, Y)
    orglist, S1list, S2list, S3list = getransusp(orgmodel, S1model, S2model, S3model, X, Y)

    # print(orgsfm)
    # print(GTsf)
    # print(S1sfm)
    # print(S2sfm)
    # print(S3sfm)
    # print(orglist)
    # print(S1list)
    # print(S2list)
    # print(S3list)

    fea = []
    for i in range(10):
        fea.append(orgsfm[0][i])
    for i in range(10):
        fea.append(GTsf[0][i])
    for i in range(10):
        fea.append(S1sfm[0][i])
    for i in range(10):
        fea.append(S2sfm[0][i])
    for i in range(10):
        fea.append(S3sfm[0][i])
    fea.append(1)
    fea.append(1)
    fea.append(1)
    fea.append(orglist[0])
    fea.append(S1list[0])
    fea.append(S2list[0])
    fea.append(S3list[0])

    # fea.append(S1list[0] + S2list[0] + S3list[0] - orglist[0])#Susp指标，现在不需要了
    fea = np.array(fea)

    if ablation == 'input':
        de = [x for x in range(30, 50)]
        de.append(51)
        de.append(52)
        de.append(55)
        de.append(56)
        fea = np.delete(fea, de, axis=0)
    elif ablation == 'hidden':
        de = [x for x in range(20, 30)]
        for i in range(40, 50):
            de.append(i)
        de.append(50)
        de.append(52)
        de.append(54)
        de.append(56)
        fea = np.delete(fea, de, axis=0)
    elif ablation == 'output':
        de = [x for x in range(20, 40)]
        de.append(50)
        de.append(51)
        de.append(54)
        de.append(55)
        fea = np.delete(fea, de, axis=0)
    elif ablation == 'input+hidden':
        de = [x for x in range(40, 50)]
        de.append(52)
        de.append(56)
        fea = np.delete(fea, de, axis=0)
    elif ablation == 'hidden+output':
        de = [x for x in range(20, 30)]
        de.append(50)
        de.append(54)
        fea = np.delete(fea, de, axis=0)
    elif ablation == 'input+output':
        de = [x for x in range(30, 40)]
        de.append(51)
        de.append(55)
        fea = np.delete(fea, de, axis=0)
    return fea


def AT(feaVec, modeltype, datatype, is_dirty, ablation):
    NUM = 100
    NUM2 = 10

    cnt = 0
    newfea = []
    tmpfea = copy.deepcopy(feaVec)
    ind = [x for x in range(tmpfea.shape[0])]
    random.shuffle(ind)  # 之前打乱错了
    tmpfea = tmpfea[ind]
    # for i in range(tmpfea.shape[0]):
    #     tmpfea[i], tmpfea[ind[i]] = tmpfea[ind[i]], tmpfea[i]

    for i in range(tmpfea.shape[0]):
        if cnt < NUM:  # and int(tmpfea[i][-1])==0:
            newfea.append(tmpfea[i][0:-2])
            cnt += 1
    for i in range(NUM2):
        newfea.append(getrandomdata(modeltype, datatype, i, ablation))

    newfea = np.array(newfea)
    # print(newfea.shape)

    Y = []
    for i in range(NUM):
        Y.append(0)
    for i in range(NUM2):
        Y.append(1)
    Y = np.array(Y)
    lg = LogisticRegression(C=1.0)
    lg.fit(newfea, Y)

    LRres = lg.predict_proba(feaVec[0:60000, 0:-2])  ####@@@@
    LRres = LRres[:, 1]

    rank = []
    for i in range(is_dirty.shape[0]):
        rank.append([int(feaVec[i][-1]), float(LRres[i])])

    rank = np.array(rank)
    rank = rank[rank[:, 1].argsort()[::-1]]
    # print(rank)
    # print("rank shape: ", rank.shape)
    save3 = RAUC(rank, bestAUC(rank))
    f = EXAM_F(rank)
    l = EXAM_L(rank)
    avg = EXAM_AVG(rank)
    save1 = PoBL(rank, 0.1)
    save2 = APFD(rank)
    return save1, save2, save3, f, l, avg


def UPDATE(feaVec, modeltype, datatype, is_dirty):
    check_ratio = 0.25
    per_check = 50
    FIRST_NUM = 100

    check_num = int((check_ratio - FIRST_NUM / feaVec.shape[0]) * (feaVec.shape[0] - FIRST_NUM))

    # 打乱
    ind = [x for x in range(feaVec.shape[0])]
    random.shuffle(ind)  # 之前打乱错了
    feaVec = feaVec[ind]
    # for i in range(feaVec.shape[0]):
    #     feaVec[i], feaVec[ind[i]] = feaVec[ind[i]], feaVec[i]

    # 第一次选择
    fea_start = feaVec[0:FIRST_NUM, :-2]
    lb_start = feaVec[0:FIRST_NUM, -1]
    print(lb_start.shape)
    if lb_start.sum() <= 2:
        newfea = []
        newlb = []
        for i in range(10):
            newfea.append(getrandomdata(modeltype, datatype, i, ''))
            newlb.append(1)
        newfea = np.array(newfea)
        newlb = np.array(newlb)
        fea_start = np.vstack((fea_start, newfea))
        lb_start = np.concatenate((lb_start, newlb))
    print('fea_start shape: ', fea_start.shape)
    print('lb_start shape: ', lb_start.shape)
    fea_left = feaVec[FIRST_NUM:, :-2]
    lb_left = feaVec[FIRST_NUM:, -1]

    lg_st = LogisticRegression(C=1.0)
    lg_st.fit(fea_start, lb_start)

    LRres = lg_st.predict_proba(fea_left)  ####@@@@
    LRres = LRres[:, 1]

    fea_left = fea_left[LRres.argsort()[::-1]]
    lb_left = lb_left[LRres.argsort()[::-1]]

    X = copy.deepcopy(fea_start)
    Y = copy.deepcopy(lb_start)
    ##开始
    for i in range(int(check_num / per_check)):
        print(str(i) + '/' + str(int(check_num / per_check)))

        fea = fea_left[0:per_check, :]
        lb = lb_left[0:per_check]

        fea_left = fea_left[per_check:, :]
        lb_left = lb_left[per_check:]

        lg = LogisticRegression(C=1.0)
        X = np.vstack((X, fea))
        Y = np.concatenate((Y, lb))
        print('X Y shape', X.shape, Y.shape)
        print('fea_left shape', fea_left.shape)
        lg.fit(X, Y)

        LRres = lg.predict_proba(fea_left)  ####@@@@
        LRres = LRres[:, 1]

        fea_left = fea_left[LRres.argsort()[::-1]]
        lb_left = lb_left[LRres.argsort()[::-1]]
        # ID = lb_left[LRres.argsort()[::-1]]

    ##结果统计
    LRres = lg.predict_proba(feaVec[:, :-2])  ####@@@@
    LRres = LRres[:, 1]

    rank = []
    for i in range(feaVec.shape[0]):
        rank.append([int(feaVec[i][-1]), float(LRres[i])])

    rank = np.array(rank)
    rank = rank[rank[:, 1].argsort()[::-1]]
    # print(rank)
    # print("rank shape: ", rank.shape)
    save3 = RAUC(rank, bestAUC(rank))
    f = EXAM_F(rank)
    l = EXAM_L(rank)
    avg = EXAM_AVG(rank)
    save1 = PoBL(rank, 0.1)
    save2 = APFD(rank)
    return save1, save2, save3, f, l, avg


def PROCESS(modeltype, datatype, ablation):
    if modeltype == 'LeNet5':
        orgmodel = LeNet5()
        S1model = getpremodel_LeNet5('./models/mnist_' + datatype + '_LeNet5.pth')
        S2model = getpremodel_LeNet5('./models/mnist_' + datatype + '_LeNet5.pth')
        S3model = getpremodel_LeNet5('./models/mnist_' + datatype + '_LeNet5.pth')
    elif modeltype == 'LeNet1':
        orgmodel = LeNet1()
        S1model = getpremodel_LeNet1('./models/mnist_' + datatype + '_LeNet1.pth')
        S2model = getpremodel_LeNet1('./models/mnist_' + datatype + '_LeNet1.pth')
        S3model = getpremodel_LeNet1('./models/mnist_' + datatype + '_LeNet1.pth')

    org_state_dict = torch.load('./models/mnist_' + datatype + '_' + modeltype + '.pth')
    orgmodel.load_state_dict(org_state_dict)

    S1_state_dict = torch.load('./retrainmodel/mnist_' + datatype + '_' + modeltype + '_retrain_VAE.pth')
    S1model.load_state_dict(S1_state_dict)

    S2_state_dict = torch.load('./retrainmodel/mnist_' + datatype + '_' + modeltype + '_retrain_Kmeans.pth')
    S2model.load_state_dict(S2_state_dict)

    S3_state_dict = torch.load('./retrainmodel/mnist_' + datatype + '_' + modeltype + '_retrain_LOSS.pth')
    S3model.load_state_dict(S3_state_dict)

    datapath = './data/MNIST/MNIST_PNG/' + datatype + 'traindata.npy'

    predata = np.load(datapath, allow_pickle=True)
    y_train = torch.from_numpy(np.array([int(x) for x in predata[:, 0]]))
    is_dirty = torch.from_numpy(np.array([int(x) for x in predata[:, 2]]))

    # orgsfm = getOUTPUT(orgmodel, datapath)
    # print('orgsfm shape:', orgsfm.shape)
    #
    # GTsfm = getGT(datapath)
    # print('GTsfm shape:', GTsfm.shape)
    #
    # S1sfm = getOUTPUT(S1model, datapath)
    # print('S1sfm shape:', S1sfm.shape)
    #
    # S2sfm = getOUTPUT(S2model, datapath)
    # print('S2sfm shape:', S2sfm.shape)
    #
    # S3sfm = getOUTPUT(S3model, datapath)
    # print('S3sfm shape:', S3sfm.shape)
    #
    # ISS1 = ISO(datapath)  # [isiso,_,_]
    # print('ISS1 shape:', ISS1.shape)
    #
    # ISS2 = clusterLeNet5(orgmodel, datapath)
    # print('ISS2 shape:', ISS2.shape)
    #
    # ISS3 = Deletedata_S1(orgmodel, datapath)  # [isdel,_,_]
    # print('ISS3 shape:', ISS3.shape)
    #
    # suslist = susp(orgmodel, S1model, S2model, S3model, datapath)
    # print('susp shape:', suslist.shape)
    #
    # orgloss, S1loss, S2loss, S3loss = Detial_susp(orgmodel, S1model, S2model, S3model, datapath)
    #
    #
    #
    # get_featureV(orgsfm, GTsfm, S1sfm, S2sfm, S3sfm, ISS1, ISS2, ISS3, suslist, orgloss, S1loss, S2loss, S3loss,
    #              is_dirty, datatype,modeltype,predata)
    # return
    # feaVec = np.load('./data/MNIST/MNIST_LR_feature/' + 'more_' + datatype + '_feature_' + modeltype + '.npy',
    #                  allow_pickle=True)#LeNet5
    # feaVec = np.load('./data/MNIST/MNIST_LR_feature/' + 'more_' + datatype + '_feature.npy',
    #                  allow_pickle=True)  # LeNet1
    feaVec = np.load('./data/MNIST/MNIST_LR_feature/' + 'new_' + datatype + '_feature_' + modeltype + '.npy',
                     allow_pickle=True)  # LeNet5
    # cls1(feaVec)

    #
    feaVec[:, -2] = feaVec[:, -2] + 2 * feaVec[:, -6]
    #
    # get_susp_rank_res(feaVec[:, -2], is_dirty)    #
    print(feaVec.shape)
    if ablation == 'input':
        de = [x for x in range(30, 50)]
        de.append(51)
        de.append(52)
        de.append(55)
        de.append(56)
        feaVec = np.delete(feaVec, de, axis=1)
    elif ablation == 'hidden':
        de = [x for x in range(20, 30)]
        for i in range(40, 50):
            de.append(i)
        de.append(50)
        de.append(52)
        de.append(54)
        de.append(56)
        feaVec = np.delete(feaVec, de, axis=1)
    elif ablation == 'output':
        de = [x for x in range(20, 40)]
        de.append(50)
        de.append(51)
        de.append(54)
        de.append(55)
        feaVec = np.delete(feaVec, de, axis=1)
    elif ablation == 'input+hidden':
        de = [x for x in range(40, 50)]
        de.append(52)
        de.append(56)
        feaVec = np.delete(feaVec, de, axis=1)
    elif ablation == 'hidden+output':
        de = [x for x in range(20, 30)]
        de.append(50)
        de.append(54)
        feaVec = np.delete(feaVec, de, axis=1)
    elif ablation == 'input+output':
        de = [x for x in range(30, 40)]
        de.append(51)
        de.append(55)
        feaVec = np.delete(feaVec, de, axis=1)

    print(feaVec.shape)
    # UPDATE(feaVec, modeltype, datatype, is_dirty)

    return AT(feaVec, modeltype, datatype, is_dirty, ablation)
    return

    num = 50
    feaVec = feaVec[feaVec[:, -2].argsort()[::-1]]

    # for i in range(50):
    #     print(feaVec[i][-2].shape)
    #     img = Image.fromarray(np.uint8(feaVec[i][-2]))
    #     print(str(i)+' '+str(feaVec[i][-1]))
    #     plt.imshow(img)
    #     plt.show()
    #     start_listen()
    # for i in range(feaVec.shape[0]-1,feaVec.shape[0]-51,-1):
    #     print(feaVec[i][-2].shape)
    #     img = Image.fromarray(np.uint8(feaVec[i][-2]))
    #     print(str(i) + ' ' + str(feaVec[i][-1]))
    #     plt.imshow(img)
    #     plt.show()
    #     start_listen()
    # print(lb_LR)

    #
    #

    X = np.vstack((feaVec[0:num, 0:-1], feaVec[-num:60000, 0:-1]))  ####@@@@
    print(X)
    # Y= np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    # lg = LogisticRegression(C=1.0)
    # lg.fit(X, Y)
    #
    # LRres = lg.predict_proba(feaVec[0:60000, 0:-3])  ####@@@@
    # LRres = LRres[:, 1]
    # rank = []
    # for i in range(feaVec.shape[0]):
    #     rank.append([int(feaVec[i][-1]), feaVec[i][-2],float(LRres[i])])
    # rank = np.array(rank)
    # rank = rank[rank[:, 2].argsort()[::-1]]
    # for i in range(rank.shape[0]):
    #
    #     img = Image.fromarray(np.uint8(rank[i][1]))
    #     print(str(i) + ' ' + str(rank[i][0]))
    #     plt.imshow(img)
    #     plt.show()

    Y = np.concatenate((feaVec[0:num, -1], feaVec[-num:60000, -1]))

    print(X.shape, Y.shape)

    lg = LogisticRegression(C=1.0)
    lg.fit(X, Y)

    LRres = lg.predict_proba(feaVec[0:60000, 0:-1])  ####@@@@
    LRres = LRres[:, 1]

    rank = []
    for i in range(is_dirty.shape[0]):
        rank.append([int(feaVec[i][-1]), float(LRres[i])])

    rank = np.array(rank)
    rank = rank[rank[:, 1].argsort()[::-1]]
    print(rank)
    print("rank shape: ", rank.shape)
    RAUC(rank, bestAUC(rank))
    EXAM_F(rank)
    EXAM_L(rank)
    EXAM_AVG(rank)
    PoBL(rank, 0.1)
    APFD(rank)

    # print(tmp)
    # for lb in range(10):
    #     cnt = 0
    #     sum = 0
    #     for i in range(y_train.shape[0]):
    #         if int(y_train[i]) == lb and int(tmp[i][0]) == 1 and int(is_dirty[i]) == 1:
    #             cnt += 1
    #         if int(y_train[i]) == lb and int(is_dirty[i]) == 1:
    #             sum += 1
    #     print(cnt, sum)
    #     if sum !=0:
    #         print(cnt/sum)


PCNT = 0
PALL = 10
from scipy.stats import ranksums
from cliffs_delta import cliffs_delta


def check(a, b):
    ans = ''
    s, p = ranksums(a, b)
    d, res = cliffs_delta(a, b)
    if p >= 0.05:
        ans = 'T'
    elif p < 0.05:
        if d >= 0.147:
            ans = 'W'
        elif d <= -0.147:
            ans = 'L'
        else:
            ans = 'T'
    return ans


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


def RQ2_1():
    # args:
    dtlist = ['alllabel', 'ranlabel', 'alldirty', 'randirty']
    datasetname = 'MNIST'
    mdlist = ['LeNet1', 'LeNet5']
    ablist = ['input', 'hidden', 'output', 'input+hidden', 'hidden+output', 'input+output', 'all']

    # res[datatype][dataratio][modeltype][args]
    res = [[[[[] for g in range(6)] for k in range(len(mdlist))] for j in range(len(ablist))] for i in
           range(len(dtlist))]

    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet('1')  # 在工作簿中新建一个表格

    for DATATYPE in dtlist:
        for ablation in ablist:
            for MODELTYPE in mdlist:
                random.seed(6657)
                for _ in range(PALL):
                    print('now run:' + DATATYPE + ' ' + ablation + ' ' + MODELTYPE + ' ' + str(_ + 1) + '/' + str(PALL))
                    t1, t2, t3, f, l, avg = PROCESS_direct(modeltype=MODELTYPE, datatype=DATATYPE, ablation=ablation,
                                                           dataratio='', pattern='AT')
                    res[dtlist.index(DATATYPE)][ablist.index(ablation)][mdlist.index(MODELTYPE)][0].append(t1)
                    res[dtlist.index(DATATYPE)][ablist.index(ablation)][mdlist.index(MODELTYPE)][1].append(t2)
                    res[dtlist.index(DATATYPE)][ablist.index(ablation)][mdlist.index(MODELTYPE)][2].append(t3)
                    res[dtlist.index(DATATYPE)][ablist.index(ablation)][mdlist.index(MODELTYPE)][3].append(f)
                    res[dtlist.index(DATATYPE)][ablist.index(ablation)][mdlist.index(MODELTYPE)][4].append(l)
                    res[dtlist.index(DATATYPE)][ablist.index(ablation)][mdlist.index(MODELTYPE)][5].append(avg)

        for MODELTYPE in mdlist:
            ORG = res[dtlist.index(DATATYPE)][ablist.index('all')][mdlist.index(MODELTYPE)]
            file = open('save.txt', 'a')
            s1 = format(sum(ORG[0]) / len(ORG[0]), '.4f')
            s2 = format(sum(ORG[1]) / len(ORG[1]), '.4f')
            s3 = format(sum(ORG[2]) / len(ORG[2]), '.4f')
            file.write(MODELTYPE + ' ' + DATATYPE + ' ' + 'all: ' + str(s1) + ' '
                       + str(s2) + ' ' + str(s3) + '\n')
            file.close()
            writexcel(sheet, mdlist.index(MODELTYPE) * len(dtlist) + dtlist.index(DATATYPE), 6, s1, '', 0)
            writexcel(sheet, mdlist.index(MODELTYPE) * len(dtlist) + dtlist.index(DATATYPE), 6, s2, '', 8)
            writexcel(sheet, mdlist.index(MODELTYPE) * len(dtlist) + dtlist.index(DATATYPE), 6, s3, '', 16)
            for ablation in ablist:
                if ablation == 'all':
                    continue
                SPC = res[dtlist.index(DATATYPE)][ablist.index(ablation)][mdlist.index(MODELTYPE)]
                PoBL_WTL = check(ORG[0], SPC[0])
                APFD_WTL = check(ORG[1], SPC[1])
                RAUC_WTL = check(ORG[2], SPC[2])
                s1 = format(sum(SPC[0]) / len(SPC[0]), '.4f')
                s2 = format(sum(SPC[1]) / len(SPC[1]), '.4f')
                s3 = format(sum(SPC[2]) / len(SPC[2]), '.4f')

                file = open('save.txt', 'a')
                file.write(MODELTYPE + ' ' + DATATYPE + ' ' + ablation + ': ' + str(s1) + ' '
                           + str(s2) + ' ' + str(s3) + ' ' + PoBL_WTL + ' ' + APFD_WTL + ' ' + RAUC_WTL + '\n')
                file.close()
                writexcel(sheet, mdlist.index(MODELTYPE) * len(dtlist) + dtlist.index(DATATYPE),
                          ablist.index(ablation), s1, PoBL_WTL, 0)
                writexcel(sheet, mdlist.index(MODELTYPE) * len(dtlist) + dtlist.index(DATATYPE),
                          ablist.index(ablation), s2, APFD_WTL, 8)
                writexcel(sheet, mdlist.index(MODELTYPE) * len(dtlist) + dtlist.index(DATATYPE),
                          ablist.index(ablation), s3, RAUC_WTL, 16)

    workbook.save('F:/ICSEdata/excel/' + datasetname + '_RQ2_1.xls')
    res = np.array(res)
    np.save('F:/ICSEdata/RQdata/' + datasetname + '_RQ2_1.npy', res)


def Online():
    dtlist = ['alllabel', 'ranlabel', 'alldirty', 'randirty']
    # args:
    MODELTYPE = 'LeNet5'

    row = -1
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet('1')  # 在工作簿中新建一个表格
    # for i in range(PALL):

    for DATATYPE in dtlist:
        row += 1
        random.seed(6657)
        t1, t2, t3, f, l, avg = PROCESS(modeltype=MODELTYPE, datatype=DATATYPE, ablation='')
        file = open('save.txt', 'a')
        s = MODELTYPE + ' ' + DATATYPE + ' ' + 'Online' + ': ' + str(t1) + ' ' + str(t2) + ' ' + str(t3) \
            + ' ' + str(f) + ' ' + str(l) + ' ' + str(
            avg) + '\n'
        file.write(s)
        file.close()
        writexcel(sheet, row, 0, t1, '', 0)
        writexcel(sheet, row, 1, t2, '', 0)
        writexcel(sheet, row, 2, t3, '', 0)
        writexcel(sheet, row, 3, f, '', 0)
        writexcel(sheet, row, 4, l, '', 0)
        writexcel(sheet, row, 5, avg, '', 0)
    workbook.save('C:/Users/WSHdeWindows/Desktop/res.xls')


def PROCESS_direct(modeltype, datatype, ablation, dataratio, pattern):
    if modeltype == 'LeNet5':
        orgmodel = LeNet5()
        S1model = LeNet5()
        S2model = LeNet5()
        S3model = LeNet5()

        orgstate_dict = torch.load('./models/mnist_' + datatype + '_LeNet5.pth')
        orgmodel.load_state_dict(orgstate_dict)

        S1state_dict = torch.load(
            './retrainmodel/mnist_' + datatype + '_LeNet5_retrain_VAE_direct' + dataratio + '.pth')
        S1model.load_state_dict(S1state_dict)

        S2state_dict = torch.load('./retrainmodel/mnist_' + datatype + '_LeNet5_retrain_Kmeans_direct.pth')
        S2model.load_state_dict(S2state_dict)

        S3state_dict = torch.load(
            './retrainmodel/mnist_' + datatype + '_LeNet5_retrain_LOSS_direct' + dataratio + '.pth')
        S3model.load_state_dict(S3state_dict)
    elif modeltype == 'LeNet1':
        orgmodel = LeNet1()
        S1model = LeNet1()
        S2model = LeNet1()
        S3model = LeNet1()

        orgstate_dict = torch.load('./models/mnist_' + datatype + '_LeNet1.pth')
        orgmodel.load_state_dict(orgstate_dict)

        S1state_dict = torch.load(
            './retrainmodel/mnist_' + datatype + '_LeNet1_retrain_VAE_direct' + dataratio + '.pth')
        S1model.load_state_dict(S1state_dict)

        S2state_dict = torch.load('./retrainmodel/mnist_' + datatype + '_LeNet1_retrain_Kmeans_direct.pth')
        S2model.load_state_dict(S2state_dict)

        S3state_dict = torch.load(
            './retrainmodel/mnist_' + datatype + '_LeNet1_retrain_LOSS_direct' + dataratio + '.pth')
        S3model.load_state_dict(S3state_dict)

    datapath = './data/MNIST/MNIST_PNG/' + datatype + 'traindata.npy'

    predata = np.load(datapath, allow_pickle=True)
    y_train = torch.from_numpy(np.array([int(x) for x in predata[:, 0]]))
    is_dirty = torch.from_numpy(np.array([int(x) for x in predata[:, 2]]))
    orgmodel.eval()
    S1model.eval()
    S2model.eval()
    S3model.eval()

    if pattern == 'getfeature':
        orgsfm = getOUTPUT(orgmodel, datapath)
        print('orgsfm shape:', orgsfm.shape)

        GTsfm = getGT(datapath)
        print('GTsfm shape:', GTsfm.shape)

        S1sfm = getOUTPUT(S1model, datapath)
        print('S1sfm shape:', S1sfm.shape)

        S2sfm = getOUTPUT(S2model, datapath)
        print('S2sfm shape:', S2sfm.shape)

        S3sfm = getOUTPUT(S3model, datapath)
        print('S3sfm shape:', S3sfm.shape)

        ISS1 = ISO(datapath)  # [isiso,_,_]
        print('ISS1 shape:', ISS1.shape)

        if modeltype == 'LeNet1':
            ISS2 = clusterLeNet1(orgmodel, datapath)
        elif modeltype == 'LeNet5':
            ISS2 = clusterLeNet5(orgmodel, datapath)

        print('ISS2 shape:', ISS2.shape)

        ISS3 = Deletedata_S1(orgmodel, datapath)  # [isdel,_,_]
        print('ISS3 shape:', ISS3.shape)

        suslist = susp(orgmodel, S1model, S2model, S3model, datapath)
        print('susp shape:', suslist.shape)

        orgloss, S1loss, S2loss, S3loss = Detial_susp(orgmodel, S1model, S2model, S3model, datapath)

        get_featureV(orgsfm, GTsfm, S1sfm, S2sfm, S3sfm, ISS1, ISS2, ISS3, suslist, orgloss, S1loss, S2loss, S3loss,
                     is_dirty, datatype, modeltype, predata, dataratio)


    elif pattern == 'AT':
        if dataratio == '':
            PP = 'direct'
        elif dataratio == '003':
            PP = '003'
        elif dataratio == '010':
            PP = '010'
        elif dataratio == '020':
            PP = '020'
        feaVec = np.load(
            './data/MNIST/MNIST_LR_feature/' + PP + '_' + datatype + '_feature_' + modeltype + '.npy',
            allow_pickle=True)  # LeNet5

        # cls1(feaVec)

        #
        feaVec[:, -2] = feaVec[:, -2] + 2 * feaVec[:, -6]

        print(feaVec.shape)
        if ablation == 'input':
            de = [x for x in range(30, 50)]
            de.append(51)
            de.append(52)
            de.append(55)
            de.append(56)
            feaVec = np.delete(feaVec, de, axis=1)
        elif ablation == 'hidden':
            de = [x for x in range(20, 30)]
            for i in range(40, 50):
                de.append(i)
            de.append(50)
            de.append(52)
            de.append(54)
            de.append(56)
            feaVec = np.delete(feaVec, de, axis=1)
        elif ablation == 'output':
            de = [x for x in range(20, 40)]
            de.append(50)
            de.append(51)
            de.append(54)
            de.append(55)
            feaVec = np.delete(feaVec, de, axis=1)
        elif ablation == 'input+hidden':
            de = [x for x in range(40, 50)]
            de.append(52)
            de.append(56)
            feaVec = np.delete(feaVec, de, axis=1)
        elif ablation == 'hidden+output':
            de = [x for x in range(20, 30)]
            de.append(50)
            de.append(54)
            feaVec = np.delete(feaVec, de, axis=1)
        elif ablation == 'input+output':
            de = [x for x in range(30, 40)]
            de.append(51)
            de.append(55)
            feaVec = np.delete(feaVec, de, axis=1)

        print(feaVec.shape)

        return AT(feaVec, modeltype, datatype, is_dirty, ablation)


def RQ2_2():
    dtlist = ['alllabel', 'ranlabel', 'alldirty', 'randirty']

    # args:
    MODELTYPE = 'LeNet5'

    row = -1
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet('1')  # 在工作簿中新建一个表格
    # for i in range(PALL):

    for DATATYPE in dtlist:
        row += 1

        save_RT = [0, 0, 0, 0, 0, 0]
        save_DT = [0, 0, 0, 0, 0, 0]

        PoBL_RT = []
        APFD_RT = []
        RAUC_RT = []

        PoBL_DT = []
        APFD_DT = []
        RAUC_DT = []
        random.seed(6657)
        for _ in range(PALL):
            print('now run:' + 'randomweight' + ' ' + DATATYPE + ' ' + str(_) + '/' + str(PALL))
            t1, t2, t3, f, l, avg = PROCESS(modeltype=MODELTYPE, datatype=DATATYPE, ablation='all')
            PoBL_RT.append(t1)
            APFD_RT.append(t2)
            RAUC_RT.append(t3)
            save_RT[0] += t1
            save_RT[1] += t2
            save_RT[2] += t3
            save_RT[3] += f
            save_RT[4] += l
            save_RT[5] += avg
        for i in range(6):
            save_RT[i] = save_RT[i] / PALL

        random.seed(6657)
        for _ in range(PALL):
            print('now run:' + 'direct' + ' ' + DATATYPE + ' ' + str(_) + '/' + str(PALL))
            t1, t2, t3, f, l, avg = PROCESS_direct(modeltype=MODELTYPE, datatype=DATATYPE, ablation='all')
            PoBL_DT.append(t1)
            APFD_DT.append(t2)
            RAUC_DT.append(t3)
            save_DT[0] += t1
            save_DT[1] += t2
            save_DT[2] += t3
            save_DT[3] += f
            save_DT[4] += l
            save_DT[5] += avg
        for i in range(6):
            save_DT[i] = save_DT[i] / PALL

        for i in range(6):
            save_RT[i] = format(save_RT[i], '.4f')
            save_DT[i] = format(save_DT[i], '.4f')

        PoBL_WTL = check(PoBL_DT, PoBL_RT)
        APFD_WTL = check(APFD_DT, APFD_RT)
        RAUC_WTL = check(RAUC_DT, RAUC_RT)

        file = open('save.txt', 'a')
        s1 = MODELTYPE + ' ' + DATATYPE + ' ' + 'direct' + ': ' + str(save_DT[0]) + ' ' + str(save_DT[1]) + ' ' + str(
            save_DT[2]) \
             + ' ' + str(save_DT[3]) + ' ' + str(save_DT[4]) + ' ' + str(
            save_DT[5]) + '\n'
        s2 = MODELTYPE + ' ' + DATATYPE + ' ' + 'randomweight' + ': ' + str(save_RT[0]) + ' ' + str(
            save_RT[1]) + ' ' + str(
            save_RT[2]) \
             + ' ' + str(save_RT[3]) + ' ' + str(save_RT[4]) + ' ' + str(
            save_RT[5]) + ' ' + PoBL_WTL + ' ' + APFD_WTL + ' ' + RAUC_WTL + '\n'
        file.write(s1)
        file.write(s2)
        file.close()

        writexcel(sheet, row, 5, save_DT[0], '', 0)
        writexcel(sheet, row, 5, save_DT[1], '', 6)
        writexcel(sheet, row, 5, save_DT[2], '', 12)

        writexcel(sheet, row, 4, save_RT[0], PoBL_WTL, 0)
        writexcel(sheet, row, 4, save_RT[1], APFD_WTL, 6)
        writexcel(sheet, row, 4, save_RT[2], RAUC_WTL, 12)
    workbook.save('C:/Users/WSHdeWindows/Desktop/res.xls')


def RQ2(dataset, datatype, modeltype, Iteration_Batchsize, remove_ratio, Mutation_Operation, Mutation_Epoch):
    feaVec = None
    indic_fea = None
    indic_gt = None
    if Mutation_Operation == 'direct':
        if remove_ratio == '005':
            feaVec = np.load('F:/ICSEdata/RQ1usedata/MNISTfeature/' + datatype + modeltype + '_feature.npy',
                             allow_pickle=True)
            new_feaVec = []

            for i in range(60000):
                # new_fea_left = fea_left[:, 0:57],fea_left[:, 60],fea_left[:, 60]
                new_feaVec.append(np.hstack((feaVec[i, 0:57], feaVec[i, 57], feaVec[i, 59])))

            new_feaVec = np.array(new_feaVec, dtype=np.float32)

            feaVec = new_feaVec

            indic_fea = -2
            indic_gt = -1

        else:
            feaVec = np.load(
                './data/MNIST/MNIST_LR_feature/' + remove_ratio + '_' + datatype + '_' + 'feature' + '_' + modeltype + '.npy',
                allow_pickle=True)
            indic_fea = -2
            indic_gt = -1

    if Mutation_Operation == 'randomweight':
        feaVec = np.load(
            './data/MNIST/MNIST_LR_feature/' + 'new_' + datatype + '_' + 'feature' + '_' + modeltype + '.npy',
            allow_pickle=True)
        indic_fea = -2
        indic_gt = -1

    print(feaVec.shape)

    # xls save path
    workbook_save_path = None
    if datatype == 'alllabel':
        workbook_save_path = 'G:/dfauloV2/RQ2/' + dataset + '/' + modeltype + '/' + Mutation_Epoch + '_' + Mutation_Operation + '_' + remove_ratio + '_' + str(
            Iteration_Batchsize) + '_' + 'RandomLabelNoise' + modeltype + '.xls'
    elif datatype == 'ranlabel':
        workbook_save_path = 'G:/dfauloV2/RQ2/' + dataset + '/' + modeltype + '/' + Mutation_Epoch + '_' + Mutation_Operation + '_' + remove_ratio + '_' + str(
            Iteration_Batchsize) + '_' + 'SpecificLabelNoise' + modeltype + '.xls'
    elif datatype == 'alldirty':
        workbook_save_path = 'G:/dfauloV2/RQ2/' + dataset + '/' + modeltype + '/' + Mutation_Epoch + '_' + Mutation_Operation + '_' + remove_ratio + '_' + str(
            Iteration_Batchsize) + '_' + 'RandomDataNoise' + modeltype + '.xls'
    elif datatype == 'randirty':
        workbook_save_path = 'G:/dfauloV2/RQ2/' + dataset + '/' + modeltype + '/' + Mutation_Epoch + '_' + Mutation_Operation + '_' + remove_ratio + '_' + str(
            Iteration_Batchsize) + '_' + 'SpecificDataNoise' + modeltype + '.xls'

    green = "font:colour_index green;"
    blue = "font:colour_index blue;"
    black = "font:colour_index black;"
    # Online
    seedd = None
    if Mutation_Epoch == '5':
        seedd = 4354
    elif Mutation_Epoch == '10':
        seedd = 6657
    elif Mutation_Epoch == '20':
        seedd = 8563

    random.seed(seedd)
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('1')
    sheet_2 = workbook.add_sheet('2')
    result = {
        'EXAM_F': [],
        'EXAM_L': [],
        'EXAM_AVG': [],
        'POBL(1)': [],
        'POBL(2)': [],
        'POBL(5)': [],
        'POBL(10)': [],
        'APFD': [],
        'RAUC': [],
        'initilize': [],
        'run_time': [],
        'epochs': [],
        'y': [],
    }

    sheet.write(0, 0, 'EXAM_F', xlwt.easyxf(blue))
    sheet.write(0, 1, 'EXAM_L', xlwt.easyxf(blue))
    sheet.write(0, 2, 'EXAM_AVG', xlwt.easyxf(blue))
    sheet.write(0, 3, 'POBL(1)', xlwt.easyxf(blue))
    sheet.write(0, 4, 'POBL(2)', xlwt.easyxf(blue))
    sheet.write(0, 5, 'POBL(5)', xlwt.easyxf(blue))
    sheet.write(0, 6, 'POBL(10)', xlwt.easyxf(blue))
    sheet.write(0, 7, 'APFD', xlwt.easyxf(blue))
    sheet.write(0, 8, 'RAUC', xlwt.easyxf(blue))
    sheet.write(0, 9, 'initilize', xlwt.easyxf(blue))
    sheet.write(0, 10, 'run_time', xlwt.easyxf(blue))
    sheet.write(0, 11, 'epochs', xlwt.easyxf(blue))

    for i in range(30):
        print('Now is ' + datatype + modeltype + '  :' + str(i))

        f, l, avg, pobl1, pobl2, pobl5, save10, apfd, rauc, y, yb, run_time, epochs, initilize = UPDATE(feaVec,
                                                                                                        modeltype,
                                                                                                        datatype,
                                                                                                        indic_fea,
                                                                                                        indic_gt,
                                                                                                        Iteration_Batchsize)

        # round(x,6)
        f, l, avg, pobl1, pobl2, pobl5, save10, apfd, rauc, run_time = round(f, 6), round(l, 6), round(avg, 6), round(
            pobl1, 6), round(pobl2, 6), round(pobl5, 6), round(save10, 6), round(apfd, 6), round(rauc, 6), round(
            run_time, 6)

        result['EXAM_F'].append(f)
        result['EXAM_L'].append(l)
        result['EXAM_AVG'].append(avg)
        result['POBL(1)'].append(pobl1)
        result['POBL(2)'].append(pobl2)
        result['POBL(5)'].append(pobl5)
        result['POBL(10)'].append(save10)
        result['APFD'].append(apfd)
        result['RAUC'].append(rauc)
        result['initilize'].append(initilize)
        result['run_time'].append(run_time)
        result['epochs'].append(epochs)
        result['y'].append(y)

        sheet.write(i + 1, 0, f, xlwt.easyxf(black))
        sheet.write(i + 1, 1, l, xlwt.easyxf(black))
        sheet.write(i + 1, 2, avg, xlwt.easyxf(black))
        sheet.write(i + 1, 3, pobl1, xlwt.easyxf(black))
        sheet.write(i + 1, 4, pobl2, xlwt.easyxf(black))
        sheet.write(i + 1, 5, pobl5, xlwt.easyxf(black))
        sheet.write(i + 1, 6, save10, xlwt.easyxf(black))
        sheet.write(i + 1, 7, apfd, xlwt.easyxf(black))
        sheet.write(i + 1, 8, rauc, xlwt.easyxf(black))
        sheet.write(i + 1, 9, initilize, xlwt.easyxf(black))
        sheet.write(i + 1, 10, run_time, xlwt.easyxf(black))
        sheet.write(i + 1, 11, epochs, xlwt.easyxf(black))

        if i == 0:
            for j in range(len(y)):
                sheet_2.write(i, j, round(yb[j], 6), xlwt.easyxf(blue))

        for j in range(len(y)):
            sheet_2.write(i + 1, j, round(y[j], 6), xlwt.easyxf(black))

        workbook.save(workbook_save_path)

    # compute the mean and std of the result
    for i, key in enumerate(result.keys()):
        if key == 'y':
            # col mean
            col_mean = np.mean(result[key], axis=0)
            col_std = np.std(result[key], axis=0)
            for j in range(len(col_mean)):
                sheet_2.write(31, j, round(col_mean[j], 6), xlwt.easyxf(green))
                sheet_2.write(32, j, round(col_std[j], 6), xlwt.easyxf(green))

        else:
            sheet.write(31, i, round(float(np.mean(result[key])), 6), xlwt.easyxf(green))
            sheet.write(32, i, round(float(np.std(result[key])), 6), xlwt.easyxf(green))

    workbook.save(workbook_save_path)


if __name__ == "__main__":
    config = [
        ['10', 'direct', '005', 200],
        ['5', 'direct', '005', 200],
        ['20', 'direct', '005', 200],
        ['10', 'randomweight', '005', 200],
        ['10', 'direct', '003', 200],
        ['10', 'direct', '010', 200],
        ['10', 'direct', '005', 100],
        ['10', 'direct', '005', 500]]

    for conf in config:
        RQ2(dataset='MNIST', datatype='ranlabel', modeltype='LeNet5', Iteration_Batchsize=conf[3],
            remove_ratio=conf[2],
            Mutation_Operation=conf[1], Mutation_Epoch=conf[0])
    # RQ2_1()
    # Online()
    # PCNT+=1
    # getrandomdata(modeltype='LeNet5', datatype='alldirty')
    # dtlist = ['alllabel', 'ranlabel', 'alldirty', 'randirty']
    # for DATATYPE in dtlist:
    #     PROCESS_direct(modeltype='LeNet5', datatype=DATATYPE, ablation='all', dataratio='020', pattern='getfeature')
