import copy
import logging
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import TensorDataset
from tqdm import tqdm
import xlwt
from torchvision import datasets, transforms
from _exp_.train_model.models import ResNet20, VGG

device = 'cuda'
from sklearn.linear_model import LogisticRegression


def ORGLABEL(datatype, modeltype):
    nowdatapath = './data/CIFA10/CIFA10_PNG/' + datatype + 'traindata_VAE_Kmeans_LOSS_' + modeltype + '.npy'
    lastdatapath = './data/CIFA10/CIFA10_PNG/' + datatype + 'traindata.npy'
    orgdatapath = './data/CIFA10/CIFA10_PNG/orgtraindata.npy'

    nowdata = np.load(nowdatapath, allow_pickle=True)
    tmpdata = np.load(lastdatapath, allow_pickle=True)
    orgdata = np.load(orgdatapath, allow_pickle=True)
    if datatype == 'alldirty' or datatype == 'randirty':
        newdata = []
        for i in range(nowdata.shape[0]):
            newdata.append([nowdata[i][0], nowdata[i][1], nowdata[i][2], nowdata[i][3], nowdata[i][4], nowdata[i][5],
                            nowdata[i][2]])
        newdata = np.array(newdata)
        print(newdata.shape)
        print('dirty')
        np.save('F:/ICSEdata/RQ1usedata/CIFAR10/' + datatype + 'traindata_VAE_Kmeans_LOSS_' + modeltype + '.npy',
                newdata)
        return
    cnt = 0
    sum = 0
    lastdata = []
    for i in range(tmpdata.shape[0]):
        if int(tmpdata[i][2]) == 1:
            lastdata.append([tmpdata[i][0], tmpdata[i][1], tmpdata[i][2], orgdata[i][0]])
    lastdata = np.array(lastdata)
    print(lastdata.shape)

    newdata = []
    for i in tqdm(range(nowdata.shape[0])):
        if int(nowdata[i][2]) == 1:
            for j in range(lastdata.shape[0]):
                if (nowdata[i][1] == lastdata[j][1]).all():
                    newdata.append(
                        [nowdata[i][0], nowdata[i][1], nowdata[i][2], nowdata[i][3], nowdata[i][4], nowdata[i][5],
                         lastdata[j][3]])
        else:
            newdata.append(
                [nowdata[i][0], nowdata[i][1], nowdata[i][2], nowdata[i][3], nowdata[i][4], nowdata[i][5],
                 nowdata[i][0]])
    newdata = np.array(newdata)
    # for i in tqdm(range(tmpdata.shape[0])):
    #     if int(tmpdata[i][2]) == 1:
    #         for j in range(newdata.shape[0]):
    #             if (orgdata[i][1]==newdata[j][1]).all() and orgdata[i][0]==newdata[j][6]:
    #                 cnt+=1
    print(newdata.shape)
    np.save('F:/ICSEdata/RQ1usedata/CIFAR10/' + datatype + 'traindata_VAE_Kmeans_LOSS_' + modeltype + '.npy', newdata)


def PoBL(ranklist, ratio):
    n = ranklist.shape[0]
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
    for i in range(ranklist.shape[0]):
        if ranklist[i] == 1:
            bestlist.append(ranklist[i])
    for i in range(ranklist.shape[0]):
        if ranklist[i] == 0:
            bestlist.append(ranklist[i])
    bestlist = np.array(bestlist)
    return bestlist


def EXAM_F(ranklist):
    n = ranklist.shape[0]
    pos = -1
    for i in range(n):
        if ranklist[i] == 1:
            pos = i
            break
    return (pos + 1) / n
    # print('EXAM_F score: ', (pos + 1) / n)


def EXAM_L(ranklist):
    n = ranklist.shape[0]
    pos = -1
    for i in range(n - 1, -1, -1):
        if ranklist[i] == 1:
            pos = i
            break
    return (pos + 1) / n
    # print('EXAM_L score: ', (pos + 1) / n)


def EXAM_AVG(ranklist):
    n = ranklist.shape[0]
    m = 0
    tf = 0
    for i in range(n):
        if ranklist[i] == 1:
            tf += i
            m += 1
    return tf / (n * m)
    # print('EXAM_AVG score: ', tf / (n * m))


def APFD(ranklist):
    n = ranklist.shape[0]
    m = 0
    tf = 0
    for i in range(n):
        if ranklist[i] == 1:
            tf += i
            m += 1

    # print('APFD score: ', 1 - (tf / (n * m)) + (1 / (2 * n)))
    return 1 - (tf / (n * m)) + (1 / (2 * n))


def getmodelout(model, X, Y):
    model.eval()
    sfout = []
    softmax_func = nn.Softmax(dim=1)

    with torch.no_grad():
        out = model(X)
        soft_output = softmax_func(out)
        sfout.append(soft_output.numpy()[0])

    sfout = np.array(sfout)
    return sfout


def getransusp(orgmodel, S1model, S2model, S3model, X, Y):
    orgmodel.eval()
    S1model.eval()
    S2model.eval()
    S3model.eval()
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失

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
    X = np.zeros((32, 32, 3))
    # print(X.shape)
    for k in range(3):
        for i in range(32):
            for j in range(32):
                X[i, j, k] = random.randint(0, 255)
    Y = lb
    X = Image.fromarray(np.uint8(X))
    if modeltype == 'ResNet':
        orgmodel = ResNet20()
        S1model = ResNet20()
        S2model = ResNet20()
        S3model = ResNet20()

        orgstate_dict = torch.load('./models/cifar10_' + datatype + '_ResNet.pth')
        orgmodel.load_state_dict(orgstate_dict)

        S1state_dict = torch.load('./retrainmodel/cifar10_' + datatype + '_ResNet_retrain_VAE.pth')
        S1model.load_state_dict(S1state_dict)

        S2state_dict = torch.load('./retrainmodel/cifar10_' + datatype + '_ResNet_retrain_Kmeans.pth')
        S2model.load_state_dict(S2state_dict)

        S3state_dict = torch.load('./retrainmodel/cifar10_' + datatype + '_ResNet_retrain_LOSS.pth')
        S3model.load_state_dict(S3state_dict)
    elif modeltype == 'VGG':
        orgmodel = VGG('VGG16')
        S1model = VGG('VGG16')
        S2model = VGG('VGG16')
        S3model = VGG('VGG16')

        orgstate_dict = torch.load('./models/cifar10_' + datatype + '_VGG.pth')
        orgmodel.load_state_dict(orgstate_dict)

        S1state_dict = torch.load('./retrainmodel/cifar10_' + datatype + '_VGG_retrain_VAE.pth')
        S1model.load_state_dict(S1state_dict)

        S2state_dict = torch.load('./retrainmodel/cifar10_' + datatype + '_VGG_retrain_Kmeans.pth')
        S2model.load_state_dict(S2state_dict)

        S3state_dict = torch.load('./retrainmodel/cifar10_' + datatype + '_VGG_retrain_LOSS.pth')
        S3model.load_state_dict(S3state_dict)

    transform_train = transforms.Compose([
        transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 用平均值和标准偏差归一化张量图像，
    ])
    X = transform_train(X)
    X = X.reshape(1, 3, 32, 32)

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

    # print(fea)
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
    random.shuffle(ind)
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
    #
    #
    rank = np.array(rank)
    rank = rank[newdata[:, 4].argsort()[::-1]]
    newdata = newdata[newdata[:, 4].argsort()[::-1]]
    # print(rank)
    # print("rank shape: ", rank.shape)
    save3, y, yb = RAUC(rank, bestAUC(rank))
    f = EXAM_F(rank)
    l = EXAM_L(rank)
    avg = EXAM_AVG(rank)
    save1 = PoBL(rank, 0.1)
    save2 = APFD(rank)
    return feaVec, save3


def FeatureExtraction(modeltype, datatype):
    def getOUTPUT(model, datapath):
        model.eval()
        transform_train = transforms.Compose([
            transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 用平均值和标准偏差归一化张量图像，
        ])
        predata = np.load(datapath, allow_pickle=True)

        x_train = torch.from_numpy(np.array([transform_train(x).numpy() for x in predata[:, 1]]))
        y_train = torch.from_numpy(np.array([int(x) for x in predata[:, 0]]))

        loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
        model.to(device)

        sfout = []
        softmax_func = nn.Softmax(dim=1)
        losslst = []

        with torch.no_grad():
            for i in tqdm(range(x_train.shape[0])):
                X = x_train[i].reshape(1, 3, 32, 32)
                X = X.to(device)

                y = y_train[i].reshape(1)
                y = y.long()
                y = y.to(device)

                output = model(X)

                soft_output = softmax_func(output)

                cur_loss = loss_fn(soft_output, y)

                soft_output = soft_output.cpu()
                cur_loss = cur_loss.cpu()

                sfout.append(soft_output.numpy()[0])
                losslst.append(cur_loss)
        losslst = np.array(losslst)
        sfout = np.array(sfout)
        return sfout, losslst

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

    def get_featureV(datapath, orgsfm, GTsfm, S1sfm, S2sfm, S3sfm, orgloss, S1loss, S2loss, S3loss, datatype,
                     modeltype):
        trainarr = np.load(datapath, allow_pickle=True)
        is_dirty = torch.from_numpy(np.array([int(x) for x in trainarr[:, 2]]))
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
            tmp.append(int(trainarr[i][3]))
            tmp.append(int(trainarr[i][4]))
            tmp.append(int(trainarr[i][5]))
            tmp.append(orgloss[i])
            tmp.append(S1loss[i])
            tmp.append(S2loss[i])
            tmp.append(S3loss[i])
            tmp.append(trainarr[i][0])  ##label
            tmp.append(trainarr[i][1])  ##data
            tmp.append(int(is_dirty[i]))  ##isdirty
            tmp.append(trainarr[i][6])  ##orglabel
            feaV.append(tmp)

        # feaV = np.array(feaV)
        # np.save('F:/ICSEdata/RQ1usedata/CIFAR10feature/' + datatype + modeltype + '_feature'  '.npy', feaV)

    path = './retrainmodel/'
    if modeltype == 'ResNet':
        orgmodel = ResNet20()
        S1model = ResNet20()
        S2model = ResNet20()
        S3model = ResNet20()

        orgstate_dict = torch.load('./models/cifar10_' + datatype + '_ResNet.pth')
        orgmodel.load_state_dict(orgstate_dict)

        S1state_dict = torch.load(path + 'cifar10_' + datatype + '_ResNet_retrain_VAE.pth')
        S1model.load_state_dict(S1state_dict)

        S2state_dict = torch.load('./retrainmodel/cifar10_' + datatype + '_ResNet_retrain_Kmeans.pth')
        S2model.load_state_dict(S2state_dict)

        S3state_dict = torch.load(path + 'cifar10_' + datatype + '_ResNet_retrain_LOSS.pth')
        S3model.load_state_dict(S3state_dict)
    elif modeltype == 'VGG':
        orgmodel = VGG('VGG16')
        S1model = VGG('VGG16')
        S2model = VGG('VGG16')
        S3model = VGG('VGG16')

        orgstate_dict = torch.load('./models/cifar10_' + datatype + '_VGG.pth')
        orgmodel.load_state_dict(orgstate_dict)

        S1state_dict = torch.load(path + 'cifar10_' + datatype + '_VGG_retrain_VAE.pth')
        S1model.load_state_dict(S1state_dict)

        S2state_dict = torch.load('./retrainmodel/cifar10_' + datatype + '_VGG_retrain_Kmeans.pth')
        S2model.load_state_dict(S2state_dict)

        S3state_dict = torch.load(path + 'cifar10_' + datatype + '_VGG_retrain_LOSS.pth')
        S3model.load_state_dict(S3state_dict)
    DP = 'F:/ICSEdata/RQ1usedata/CIFAR10/'
    datapath = DP + datatype + 'traindata_VAE_Kmeans_LOSS_' + modeltype + '.npy'
    predata = np.load(datapath, allow_pickle=True)
    # y_train = torch.from_numpy(np.array([int(x) for x in predata[:, 0]]))
    is_dirty = torch.from_numpy(np.array([int(x) for x in predata[:, 2]]))
    orgmodel.eval()
    S1model.eval()
    S2model.eval()
    S3model.eval()

    start1=time.time()
    orgsfm, orgloss = getOUTPUT(orgmodel, datapath)
    print('org shape:', orgsfm.shape, orgloss.shape)

    GTsfm = getGT(datapath)
    print('GTsfm shape:', GTsfm.shape)

    S1sfm, S1loss = getOUTPUT(S1model, datapath)
    print('S1 shape:', S1sfm.shape, S1loss.shape)

    S2sfm, S2loss = getOUTPUT(S2model, datapath)
    print('S2 shape:', S2sfm.shape, S2loss.shape)

    S3sfm, S3loss = getOUTPUT(S3model, datapath)
    print('S3 shape:', S3sfm.shape, S3loss.shape)

    # suslist = S1loss + S2loss + S3loss - orgloss

    get_featureV(datapath, orgsfm, GTsfm, S1sfm, S2sfm, S3sfm, orgloss, S1loss, S2loss, S3loss, datatype,
                 modeltype)
    end1=time.time()
    print("运行时间为：{}".format(end1 - start1))


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

def Online_getAPFD_RAUC(fea):  # 极致优化
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s -  %(message)s')
logger = logging.getLogger(__name__)
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
    end1=time.time()
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
    np.save('F:\\ICSEdata\\online_new\\CIFAR10\\' + datatype + '_' + modeltype + '_online.npy', newdata)

    y = np.array(y)
    EPOCHlist = np.array(EPOCHlist)
    APFDlist = np.array(APFDlist)
    RAUClist = np.array(RAUClist)

    np.save('F:\\ICSEdata\\online_new\\CIFAR10\\' + datatype + '_' + modeltype + '_online_Cord.npy', y)
    np.save('F:\\ICSEdata\\online_new\\CIFAR10\\' + datatype + '_' + modeltype + '_EPOCH_Cord.npy', EPOCHlist)
    np.save('F:\\ICSEdata\\online_new\\CIFAR10\\' + datatype + '_' + modeltype + '_APFD_Cord.npy', APFDlist)
    np.save('F:\\ICSEdata\\online_new\\CIFAR10\\' + datatype + '_' + modeltype + '_RAUC_Cord.npy', RAUClist)

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
    return save1, save2, save3, f, l, avg


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


import cleanlab

from CL import CLEANLAB_CV
def CleanLab(datatype,modeltype):
    return CLEANLAB_CV(datatype,modeltype)


import torch.nn.functional as F


def Uncertainty(feaVec, datatype, modeltype):
    if modeltype == 'ResNet':
        model = ResNet20()
        orgstate_dict = torch.load('./models/cifar10_' + datatype + '_ResNet.pth')
        model.load_state_dict(orgstate_dict)

    elif modeltype == 'VGG':
        model = VGG('VGG16')
        orgstate_dict = torch.load('./models/cifar10_' + datatype + '_VGG.pth')
        model.load_state_dict(orgstate_dict)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    transform_train = transforms.Compose([
        transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 用平均值和标准偏差归一化张量图像，

    ])
    x_train = torch.from_numpy(np.array([transform_train(x).numpy() for x in feaVec[:, -3]]))
    y_train = torch.from_numpy(np.array([int(x) for x in feaVec[:, -4]]))
    traindataset = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=128)
    T = 20
    unct_list = []

    with torch.no_grad():
        for i, (X, y) in tqdm(enumerate(train_loader)):
            output_list = []
            for j in range(T):
                output_list.append(torch.unsqueeze(F.softmax(model(X), dim=1), dim=0))
            for j in range(X.shape[0]):
                sub_put = []
                for k in range(T):
                    sub_put.append(output_list[k][0][j])
                output_variance = torch.cat(sub_put, 0).var(dim=0).mean().item()
                unct_list.append(output_variance)
    print(len(unct_list))
    rank = []
    for i in range(len(unct_list)):
        rank.append([feaVec[i][-2], unct_list[i]])
    rank = np.array(rank)

    newdata = []
    for i in range(feaVec.shape[0]):
        newdata.append([feaVec[i][-4], feaVec[i][-3], feaVec[i][-2], feaVec[i][-1], float(unct_list[i])])
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


def RQ1(dataset, datatype, modeltype, methodology):
    feaVec = np.load('F:/ICSEdata/RQ1usedata/CIFAR10feature/' + datatype + modeltype + '_feature.npy',
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
    # Online
    # random.seed(6657)
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
        elif methodology == 'uncertainty':
            f, l, avg, pobl1, pobl2, pobl5, save10, apfd, rauc, y, yb, run_time = Uncertainty(feaVec, datatype,
                                                                                              modeltype)
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

def NEW_ONLINE(datatype, modeltype, excelx, sheet):
    feaVec = np.load('F:/ICSEdata/RQ1usedata/CIFAR10feature/' + datatype + modeltype + '_feature.npy',
                     allow_pickle=True)

    t1, t2, t3, f, l, avg = UPDATE(feaVec, modeltype, datatype)
    writexcel(sheet, excelx, 1, f, '', 0)
    writexcel(sheet, excelx, 1, l, '', 6)
    writexcel(sheet, excelx, 1, avg, '', 12)
    writexcel(sheet, excelx + 29, 1, t1, '', 0)
    writexcel(sheet, excelx + 29, 1, t2, '', 6)
    writexcel(sheet, excelx + 29, 1, t3, '', 12)
    return sheet
import warnings

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    dtlist = ['alllabel', 'ranlabel', 'alldirty', 'randirty']
    mdlist = ['ResNet']
    for dt in dtlist:
        for md in mdlist:
            RQ1('CIFAR10', dt, md, 'cleanlab')

    # FeatureExtraction('VGG','alllabel')
    # dtlist = ['alllabel', 'ranlabel', 'alldirty', 'randirty']
    # mdlist = ['ResNet', 'VGG']
    # workbook = xlwt.Workbook()  # 新建一个工作簿
    # sheet = workbook.add_sheet('1')  # 在工作簿中新建一个表格
    # row = -1
    # for MODELTYPE in mdlist:
    #     for DATATYPE in dtlist:
    #         row += 1
    #         logger.info('now run:'+MODELTYPE+' '+DATATYPE)
    #         sheet = NEW_ONLINE(DATATYPE, MODELTYPE, row, sheet)
    #         workbook.save('F:\\ICSEdata\\online_new\\CIFAR10\\CIFAR10_RQ1.xls')

    # dtlist = ['alllabel', 'ranlabel', 'alldirty', 'randirty']
    # datasetname = 'CIFAR10'
    # mdlist = ['ResNet', 'VGG']
    # workbook = xlwt.Workbook()  # 新建一个工作簿
    # sheet = workbook.add_sheet('1')  # 在工作簿中新建一个表格
    # row = -1
    # for MODELTYPE in mdlist:
    #     for DATATYPE in dtlist:
    #         row += 1
    #         sheet = RQ1(MODELTYPE, DATATYPE, row, sheet)
    # workbook.save('F:/ICSEdata/excel/' + datasetname + '_RQ1.xls')
