import copy
import logging
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import xlwt
from PIL import Image, ImageDraw
from pyod.models.iforest import IForest
from sklearn.cluster import KMeans
from torch import nn
from tqdm import tqdm
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
import cleanlab
from _exp_.train_model.models import LeNet5, LeNet1


def PoBL(ranklist, ratio):
    n = 60000
    m = 0
    for i in range(n):
        if ranklist[i] == 1:
            m += 1
    cnt = 0
    for i in range(int(n * ratio)):
        if ranklist[i] == 1:
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
    return np.trapz(y, rat) / np.trapz(yb, rat), y, yb


def bestAUC(ranklist):
    bestlist = []
    for i in range(60000):
        if ranklist[i] == 1:
            bestlist.append(ranklist[i])
    for i in range(60000):
        if ranklist[i] == 0:
            bestlist.append(ranklist[i])
    bestlist = np.array(bestlist)
    return bestlist


def EXAM_F(ranklist):
    n = 60000
    pos = -1
    for i in range(n):
        if ranklist[i] == 1:
            pos = i
            break
    return (pos + 1) / n
    # print('EXAM_F score: ', (pos + 1) / n)


def EXAM_L(ranklist):
    n = 60000
    pos = -1
    for i in range(n - 1, -1, -1):
        if ranklist[i] == 1:
            pos = i
            break
    return (pos + 1) / n
    # print('EXAM_L score: ', (pos + 1) / n)


def EXAM_AVG(ranklist):
    n = 60000
    m = 0
    tf = 0
    for i in range(n):
        if ranklist[i] == 1:
            tf += i
            m += 1
    return tf / (n * m)
    # print('EXAM_AVG score: ', tf / (n * m))


def APFD(ranklist):
    n = 60000
    m = 0
    tf = 0
    for i in range(n):
        if ranklist[i] == 1:
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


def ORGLABEL(datatype, newdatasavepath):
    nowdatapath = './data/MNIST/MNIST_PNG/' + datatype + 'traindata.npy'
    orgdatapath = './data/MNIST/MNIST_PNG/orgtraindata.npy'

    nowdata = np.load(nowdatapath, allow_pickle=True)
    orgdata = np.load(orgdatapath, allow_pickle=True)
    cnt = 0
    sum = 0
    cnt2 = 0
    newdata = []  # [label,picture,isdirty,orglabel]
    if datatype == 'alllabel' or datatype == 'ranlabel':
        for i in tqdm(range(nowdata.shape[0])):
            if int(nowdata[i][2]) == 1 and (nowdata[i][1] == orgdata[i][1]).all():
                cnt += 1
                newdata.append([nowdata[i][0], nowdata[i][1], nowdata[i][2], orgdata[i][0]])
            elif int(nowdata[i][2]) == 0:
                newdata.append([nowdata[i][0], nowdata[i][1], nowdata[i][2], orgdata[i][0]])
                if nowdata[i][0] == orgdata[i][0]:
                    cnt2 += 1
            if int(nowdata[i][2]) == 1:
                sum += 1
    if datatype == 'alldirty' or datatype == 'randirty':
        for i in tqdm(range(nowdata.shape[0])):
            if int(nowdata[i][2]) == 1:
                cnt += 1
                newdata.append([nowdata[i][0], nowdata[i][1], nowdata[i][2], 1])
            elif int(nowdata[i][2]) == 0:
                newdata.append([nowdata[i][0], nowdata[i][1], nowdata[i][2], 0])
                if nowdata[i][0] == orgdata[i][0]:
                    cnt2 += 1
            if int(nowdata[i][2]) == 1:
                sum += 1
    newdata = np.array(newdata)
    print(cnt, sum, cnt2)
    print(newdata.shape)
    np.save(newdatasavepath, newdata)


def FeatureExtraction(modeltype, datatype):
    if modeltype == 'LeNet5':
        orgmodel = LeNet5()
        S1model = LeNet5()
        S2model = LeNet5()
        S3model = LeNet5()

        orgstate_dict = torch.load('./models/mnist_' + datatype + '_LeNet5.pth')
        orgmodel.load_state_dict(orgstate_dict)

        S1state_dict = torch.load('./retrainmodel/mnist_' + datatype + '_LeNet5_retrain_VAE_direct.pth')
        S1model.load_state_dict(S1state_dict)

        S2state_dict = torch.load('./retrainmodel/mnist_' + datatype + '_LeNet5_retrain_Kmeans_direct.pth')
        S2model.load_state_dict(S2state_dict)

        S3state_dict = torch.load('./retrainmodel/mnist_' + datatype + '_LeNet5_retrain_LOSS_direct.pth')
        S3model.load_state_dict(S3state_dict)
    elif modeltype == 'LeNet1':
        orgmodel = LeNet1()
        S1model = LeNet1()
        S2model = LeNet1()
        S3model = LeNet1()

        orgstate_dict = torch.load('./models/mnist_' + datatype + '_LeNet1.pth')
        orgmodel.load_state_dict(orgstate_dict)

        S1state_dict = torch.load('./retrainmodel/mnist_' + datatype + '_LeNet1_retrain_VAE_direct.pth')
        S1model.load_state_dict(S1state_dict)

        S2state_dict = torch.load('./retrainmodel/mnist_' + datatype + '_LeNet1_retrain_Kmeans_direct.pth')
        S2model.load_state_dict(S2state_dict)

        S3state_dict = torch.load('./retrainmodel/mnist_' + datatype + '_LeNet1_retrain_LOSS_direct.pth')
        S3model.load_state_dict(S3state_dict)
    datapath = 'F:/ICSEdata/RQ1usedata/MNIST/' + datatype + 'traindata.npy'
    predata = np.load(datapath, allow_pickle=True)
    y_train = torch.from_numpy(np.array([int(x) for x in predata[:, 0]]))
    is_dirty = torch.from_numpy(np.array([int(x) for x in predata[:, 2]]))
    orgmodel.eval()
    S1model.eval()
    S2model.eval()
    S3model.eval()

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

    def get_featureV(orgsfm, GTsfm, S1sfm, S2sfm, S3sfm, ISS1, ISS2, ISS3, orgloss, S1loss, S2loss, S3loss,
                     is_dirty, datatype, modeltype, predata):
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
            tmp.append(predata[i][0])  ##label
            tmp.append(predata[i][1])  ##picture
            tmp.append(int(is_dirty[i]))  ##isdirty
            tmp.append(predata[i][3])  ##orglabel
            feaV.append(tmp)

        feaV = np.array(feaV)

        # np.save('F:/ICSEdata/RQ1usedata/MNISTfeature/' + datatype + modeltype + '_feature'  '.npy', feaV)

    start1 = time.time()
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

    end1 = time.time()

    ISS1 = ISO(datapath)  # [isiso,_,_]
    print('ISS1 shape:', ISS1.shape)

    if modeltype == 'LeNet1':
        ISS2 = clusterLeNet1(orgmodel, datapath)
    elif modeltype == 'LeNet5':
        ISS2 = clusterLeNet5(orgmodel, datapath)

    print('ISS2 shape:', ISS2.shape)

    ISS3 = Deletedata_S1(orgmodel, datapath)  # [isdel,_,_]
    print('ISS3 shape:', ISS3.shape)

    # suslist = susp(orgmodel, S1model, S2model, S3model, datapath)
    # print('susp shape:', suslist.shape)

    start2 = time.time()
    orgloss, S1loss, S2loss, S3loss = Detial_susp(orgmodel, S1model, S2model, S3model, datapath)

    get_featureV(orgsfm, GTsfm, S1sfm, S2sfm, S3sfm, ISS1, ISS2, ISS3, orgloss, S1loss, S2loss, S3loss,
                 is_dirty, datatype, modeltype, predata)
    end2 = time.time()

    print("运行时间为：{}".format((end1 - start1) + (end2 - start2)))


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


def getrandomdata(modeltype, datatype, lb):
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

    fea = np.array(fea)
    return fea


def AT(feaVec, modeltype, datatype):
    isdt = np.array([int(x) for x in feaVec[:, -2]])
    feaVecsimple = feaVec[:, 0:-4]
    NUM = 100
    NUM2 = 10

    cnt = 0
    newfea = []
    tmpfea = copy.deepcopy(feaVecsimple)
    ind = [x for x in range(tmpfea.shape[0])]
    random.shuffle(ind)  # 之前打乱错了
    tmpfea = tmpfea[ind]
    # for i in range(tmpfea.shape[0]):
    #     tmpfea[i], tmpfea[ind[i]] = tmpfea[ind[i]], tmpfea[i]
    start = time.time()
    for i in range(tmpfea.shape[0]):
        if cnt < NUM:  # and int(tmpfea[i][-1])==0:
            newfea.append(tmpfea[i])
            cnt += 1
    for i in range(NUM2):
        newfea.append(getrandomdata(modeltype, datatype, i))

    newfea = np.array(newfea)
    # print(newfea.shape)

    Y = []
    for i in range(NUM):
        Y.append(0)
    for i in range(NUM2):
        Y.append(1)
    Y = np.array(Y)
    print('start')
    lg = LogisticRegression(C=1.0)
    lg.fit(newfea, Y)
    print('finish')
    LRres = lg.predict_proba(feaVecsimple)  ####@@@@
    LRres = LRres[:, 1]

    rank = []
    for i in range(isdt.shape[0]):
        rank.append(int(isdt[i]))
    newdata = []
    for i in range(isdt.shape[0]):
        newdata.append([feaVec[i][-4], feaVec[i][-3], feaVec[i][-2], feaVec[i][-1], float(LRres[i])])
    end = time.time()
    print("运行时间为：{}".format(end - start))
    newdata = np.array(newdata)
    feaVec = feaVec[newdata[:, 4].argsort()[::-1]]

    rank = np.array(rank)
    rank = rank[newdata[:, 4].argsort()[::-1]]
    # print(rank)
    # print("rank shape: ", rank.shape)
    save3, y, yb = RAUC(rank, bestAUC(rank))
    f = EXAM_F(rank)
    l = EXAM_L(rank)
    avg = EXAM_AVG(rank)
    save1 = PoBL(rank, 0.1)
    save2 = APFD(rank)
    return feaVec, save3
    newdata = newdata[newdata[:, 4].argsort()[::-1]]

    return save1, save2, save3, f, l, avg, y, yb, newdata


logging.basicConfig(level=logging.INFO, format='%(asctime)s -  %(message)s')
logger = logging.getLogger(__name__)


def Online_getAPFD_RAUC(fea):
    rank = fea[:, -2].astype('int')
    NUM_DIRTY = rank.sum()
    NUM_ALL = rank.shape[0]
    m = 0
    tf = 0
    POBL_ = []
    BEST_POBL_ = []
    for i in range(101):
        bs = float(int((i / 100.) * NUM_ALL) / NUM_DIRTY)
        if bs > 1.:
            bs = 1.
        BEST_POBL_.append(bs * 100)
    ckp = [int((_ / 100.) * NUM_ALL) for _ in range(101)]  # 前缀和优化
    ind = 0
    noise_sum = 0
    for i in range(NUM_ALL):
        if ckp[ind] == i:
            POBL_.append(float(noise_sum / NUM_DIRTY) * 100)
            ind += 1
        if rank[i] == 1:
            noise_sum += 1
            tf += i
            m += 1
    POBL_.append(float(noise_sum / NUM_DIRTY) * 100)

    rat = [x for x in range(101)]
    rauc = np.trapz(POBL_, rat) / np.trapz(BEST_POBL_, rat)
    apfd = 1 - (tf / (NUM_ALL * m)) + (1 / (2 * NUM_ALL))

    return apfd, rauc


def UPDATE(feaVec, modeltype, datatype):
    N = feaVec.shape[0]  # data num
    # args###################################################
    check_ratio = 0.25
    per_check = 200
    epochs = int((check_ratio * N) / per_check)
    logger.info('[datanum,check_ratio,per_check,epochs]=' + str([N, check_ratio, per_check, epochs]))
    ########################################################

    # offline part#######

    la = -1
    fea_left, sa = AT(feaVec, modeltype, datatype)
    logger.info('success get offline result shape:' + str(fea_left.shape))

    logger.info('start online')
    start1 = time.time()
    global X
    APFDlist = []
    RAUClist = []
    EPOCHlist = []

    for i in range(epochs):
        EPOCHlist.append(i)
        logger.info('run: ' + str(i + 1) + '/' + str(epochs) + ' epoch')

        fea = fea_left[0:per_check, :]

        fea_left = fea_left[per_check:, :]

        lg = LogisticRegression(C=1.0)
        if i == 0:
            X = fea[:]
        else:
            X = np.vstack((X, fea))

        logger.info('X shape: ' + str(X.shape))
        logger.info('fea_left shape: ' + str(fea_left.shape))
        lg.fit(X[:, 0:-4].astype('float32'), X[:, -2].astype('int'))

        LRres = lg.predict_proba(fea_left[:, 0:-4].astype('float32'))  # 预测剩余数据
        total_res = lg.predict_proba(feaVec[:, 0:-4].astype('float32'))  # 预测整个数据集

        LRres = LRres[:, 1]
        total_res = total_res[:, 1]

        fea_left = fea_left[LRres.argsort()[::-1]]  # 根据预测结果排序*剩余*数据
        feaVec = feaVec[total_res.argsort()[::-1]]  # 根据预测结果排序*整个*数据集

        apfd, rauc = Online_getAPFD_RAUC(feaVec)
        APFDlist.append(apfd)
        RAUClist.append(rauc)

        if i == epochs - 1:
            X = np.vstack((X, fea_left))
    X = np.array(X)
    logger.info('final X shape: ' + str(X.shape))
    end1 = time.time()
    print("运行时间为：{}".format(end1 - start1))
    return
    rank = X[:, -2]
    save3, y, yb = RAUC(rank, bestAUC(rank))
    f = EXAM_F(rank)
    l = EXAM_L(rank)
    avg = EXAM_AVG(rank)
    save1 = PoBL(rank, 0.1)
    save2 = APFD(rank)
    newdata = []
    for i in range(X.shape[0]):
        newdata.append([X[i][-4], X[i][-3], X[i][-2], X[i][-1]])
    newdata = np.array(newdata)
    np.save('F:\\ICSEdata\\online_new\\MNIST\\' + datatype + '_' + modeltype + '_online.npy', newdata)

    y = np.array(y)
    EPOCHlist = np.array(EPOCHlist)
    APFDlist = np.array(APFDlist)
    RAUClist = np.array(RAUClist)

    np.save('F:\\ICSEdata\\online_new\\MNIST\\' + datatype + '_' + modeltype + '_online_Cord.npy', y)
    np.save('F:\\ICSEdata\\online_new\\MNIST\\' + datatype + '_' + modeltype + '_EPOCH_Cord.npy', EPOCHlist)
    np.save('F:\\ICSEdata\\online_new\\MNIST\\' + datatype + '_' + modeltype + '_APFD_Cord.npy', APFDlist)
    np.save('F:\\ICSEdata\\online_new\\MNIST\\' + datatype + '_' + modeltype + '_RAUC_Cord.npy', RAUClist)

    X = [x for x in range(101)]

    plt.figure()
    plt.title(datatype + ' ' + modeltype)
    plt.plot(X, yb, color='blue', label='Theory Best')
    plt.plot(X, y, color='red', label='Online')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title(datatype + ' ' + modeltype)
    plt.plot(EPOCHlist, APFDlist, color='Aqua', label='APFD')
    plt.plot(EPOCHlist, RAUClist, color='red', label='RAUC')
    plt.legend()
    plt.show()
    print(save1, save2, save3, f, l, avg)
    return save1, save2, save3, f, l, avg


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


def DeepGini(feaVec, numsfm):
    rank = []
    for i in range(feaVec.shape[0]):
        gini = 0
        for j in range(numsfm):
            gini += feaVec[i][j] ** 2
        gini = 1 - gini
        rank.append([feaVec[i, -2], gini])
    rank = np.array(rank)

    newdata = []
    for i in range(feaVec.shape[0]):
        newdata.append([feaVec[i][-4], feaVec[i][-3], feaVec[i][-2], feaVec[i][-1], float(rank[i, 1])])
    newdata = np.array(newdata)
    newdata = newdata[newdata[:, 4].argsort()[::-1]]

    rank = rank[rank[:, 1].argsort()[::-1]]
    # print(rank)
    # print("rank shape: ", rank.shape)
    save3, y, yb = RAUC(rank, bestAUC(rank))
    f = EXAM_F(rank)
    l = EXAM_L(rank)
    avg = EXAM_AVG(rank)
    save1 = PoBL(rank, 0.1)
    save2 = APFD(rank)
    return save1, save2, save3, f, l, avg, y, yb, newdata


from CL import CLEANLAB_CV


def CleanLab(datatype, modeltype):
    return CLEANLAB_CV(datatype, modeltype)


def RQ1(dataset, datatype, modeltype, methodology):
    feaVec = np.load('F:/ICSEdata/RQ1usedata/MNISTfeature/' + datatype + modeltype + '_feature.npy',
                     allow_pickle=True)  # LeNet5
    # xls save path
    workbook_save_path = None
    if datatype == 'alllabel':
        workbook_save_path = 'G:/dfauloV2/RQ1/' + methodology + '/' + dataset + '/RandomLabelNoise' + modeltype + '.xls'
    elif datatype == 'ranlabel':
        workbook_save_path = 'G:/dfauloV2/RQ1/' + methodology + '/' + dataset + '/SpecificLabelNoise' + modeltype + '.xls'
    elif datatype == 'alldirty':
        workbook_save_path = 'G:/dfauloV2/RQ1/' + methodology + '/' + dataset + '/RandomDataNoise' + modeltype + '.xls'
    elif datatype == 'randirty':
        workbook_save_path = 'G:/dfauloV2/RQ1/' + methodology + '/' + dataset + '/SpecificDataNoise' + modeltype + '.xls'
    green = "font:colour_index green;"
    blue = "font:colour_index blue;"
    black = "font:colour_index black;"

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
        'run_time': [],
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
    sheet.write(0, 9, 'run_time', xlwt.easyxf(blue))

    for i in range(3):
        print('Now is ' + datatype + modeltype + '  :' + str(i))
        if methodology == 'ours':
            f, l, avg, pobl1, pobl2, pobl5, save10, apfd, rauc, y, yb, run_time = UPDATE(feaVec, modeltype, datatype)
        elif methodology == 'deepgini':
            f, l, avg, pobl1, pobl2, pobl5, save10, apfd, rauc, y, yb, run_time = DeepGini(feaVec, 10)
        elif methodology == 'cleanlab':
            f, l, avg, pobl1, pobl2, pobl5, save10, apfd, rauc, y, yb, run_time = CleanLab(feaVec, 10)
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
        result['run_time'].append(run_time)
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
        sheet.write(i + 1, 9, run_time, xlwt.easyxf(black))

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
                sheet_2.write(21, j, round(col_mean[j], 6), xlwt.easyxf(green))
                sheet_2.write(22, j, round(col_std[j], 6), xlwt.easyxf(green))

        else:
            sheet.write(21, i, round(float(np.mean(result[key])), 6), xlwt.easyxf(green))
            sheet.write(22, i, round(float(np.std(result[key])), 6), xlwt.easyxf(green))

    workbook.save(workbook_save_path)


import warnings


# warnings.filterwarnings("ignore")
def NEW_ONLINE(datatype, modeltype, excelx, sheet):
    feaVec = np.load('F:/ICSEdata/RQ1usedata/MNISTfeature/' + datatype + modeltype + '_feature.npy',
                     allow_pickle=True)  # LeNet5

    t1, t2, t3, f, l, avg = UPDATE(feaVec, modeltype, datatype)
    # writexcel(sheet, excelx, 1, f, '', 0)
    # writexcel(sheet, excelx, 1, l, '', 6)
    # writexcel(sheet, excelx, 1, avg, '', 12)
    # writexcel(sheet, excelx + 29, 1, t1, '', 0)
    # writexcel(sheet, excelx + 29, 1, t2, '', 6)
    # writexcel(sheet, excelx + 29, 1, t3, '', 12)
    # return sheet


import warnings

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    dtlist = ['alllabel', 'ranlabel', 'alldirty', 'randirty']
    for dt in dtlist:
        RQ1(dataset='MNIST', datatype=dt, modeltype='LeNet1', methodology='cleanlab')

    # NEW_ONLINE('ranlabel', 'LeNet5', '', '')
    # dtlist = ['alllabel', 'ranlabel', 'alldirty', 'randirty']
    # mdlist = ['LeNet1', 'LeNet5']
    # workbook = xlwt.Workbook()  # 新建一个工作簿
    # sheet = workbook.add_sheet('1')  # 在工作簿中新建一个表格
    # row = -1
    # for MODELTYPE in mdlist:
    #     for DATATYPE in dtlist:
    #         row += 1
    #         logger.info('now run:'+MODELTYPE+' '+DATATYPE)
    #         sheet = NEW_ONLINE(DATATYPE, MODELTYPE, row, sheet)
    #         workbook.save('F:\\ICSEdata\\online_new\\MNIST\\MNIST_RQ1.xls')

    # args
    # NEW_ONLINE('alllabel', 'LeNet5')
    # dtlist = ['alllabel', 'ranlabel', 'alldirty', 'randirty']
    # datasetname = 'MNIST'
    # mdlist = ['LeNet1', 'LeNet5']
    #
    # workbook = xlwt.Workbook()  # 新建一个工作簿
    # sheet = workbook.add_sheet('1')  # 在工作簿中新建一个表格
    # row = -1
    #
    # for MODELTYPE in mdlist:
    #     for DATATYPE in dtlist:
    #         row += 1
    #         sheet = RQ1(MODELTYPE, DATATYPE, row, sheet)
    # workbook.save('F:/ICSEdata/excel/' + datasetname + '_RQ1.xls')

    # ORGLABEL(datatype,'F:/ICSEdata/RQ1usedata/MNIST/'+datatype+'traindata.npy')
    # FeatureExtraction('LeNet5','randirty')
