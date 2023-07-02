import copy
import logging
import pickle
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import xlwt
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.autograd.grad_mode import F
from torch.utils.data import TensorDataset
from tqdm import tqdm

from torchtext.data.utils import get_tokenizer
from _exp_.train_model.models import TCDCNN
from os.path import isfile, join


def ORGLABEL(datatype):
    orgdatapath = '../MTFL/training.txt'
    lastdatapath = '../MTFL/training_' + datatype + '.txt'

    nowdatapath = '../MTFL/TCDCN_NEWDATA_OUT/training_' + datatype + '_VAE_MID_OUT_new.txt'
    nowbbpath = '../MTFL/TCDCN_NEWDATA_OUT/annotation_' + datatype + '_VAE_MID_OUT_new.txt'

    orgdata = np.loadtxt(orgdatapath, dtype=str)
    tmpdata = np.loadtxt(lastdatapath, dtype=str)
    nowdata = np.loadtxt(nowdatapath, dtype=str)
    nowbb = np.loadtxt(nowbbpath, dtype=str)

    sum = 0
    newdatatxt = open('../MTFL/TCDCN_NEWDATA_OUT/training_' + datatype + '_VAE_MID_OUT_new_hol.txt', 'w+')
    for i in tqdm(range(nowdata.shape[0])):
        if int(nowdata[i][-4]) == 1:
            for j in range(orgdata.shape[0]):
                if orgdata[j][0] == nowdata[i][0]:
                    s = ''
                    for tmpstr in nowdata[i]:
                        s += tmpstr + ' '
                    for k in range(1, 11):
                        s += orgdata[j][k] + ' '
                    s += '\n'
                    newdatatxt.write(s)
        elif int(nowdata[i][-4]) == 0:
            s = ''
            for tmpstr in nowdata[i]:
                s += tmpstr + ' '
            for k in range(1, 11):
                s += nowdata[i][k] + ' '
            s += '\n'
            newdatatxt.write(s)
    newdatatxt.close()


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


def FeatureExtraction(datatype):
    def getOUTPUT(model, imageset, lm):
        model.eval()
        sfout = []
        losslst = []
        with torch.no_grad():
            for i, img in tqdm(enumerate(imageset)):
                img = torch.from_numpy(img)
                out = model(img.float())
                landmark = lm[i].reshape(1, -1)

                loss = model.loss([out],
                                  [landmark.float()])
                loss = loss.numpy()
                sfout.append(out.numpy()[0])
                losslst.append(loss)

        sfout = np.array(sfout)
        losslst = np.array(losslst)
        return sfout, losslst

    def get_featureV(data, orgsfm, GTsfm, S1sfm, S2sfm, S3sfm, orgloss, S1loss, S2loss, S3loss, suslist, datatype):
        feaV = []
        for i in range(data.shape[0]):
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
            tmp.append(int(data[i][15]))#17
            tmp.append(int(data[i][16]))#18
            tmp.append(int(data[i][17]))#19
            tmp.append(orgloss[i])
            tmp.append(S1loss[i])
            tmp.append(S2loss[i])
            tmp.append(S3loss[i])
            tmp.append(float(data[i][1]))  ##label57:78
            tmp.append(float(data[i][2]))
            tmp.append(float(data[i][3]))
            tmp.append(float(data[i][4]))
            tmp.append(float(data[i][5]))
            tmp.append(float(data[i][6]))
            tmp.append(float(data[i][7]))
            tmp.append(float(data[i][8]))
            tmp.append(float(data[i][9]))
            tmp.append(float(data[i][10]))
            tmp.append(data[i][0])  ##data
            tmp.append(int(0))  ##isdirty  data[i][15]
            if datatype == 'D1':
                tmp.append(float(data[i][19]))  ##orglabel
                tmp.append(float(data[i][20]))
                tmp.append(float(data[i][21]))
                tmp.append(float(data[i][22]))
                tmp.append(float(data[i][23]))
                tmp.append(float(data[i][24]))
                tmp.append(float(data[i][25]))
                tmp.append(float(data[i][26]))
                tmp.append(float(data[i][27]))
                tmp.append(float(data[i][28]))
            elif datatype == 'D2':
                tmp.append(float(data[i][1]))  ##orglabel为了保持一致
                tmp.append(float(data[i][2]))
                tmp.append(float(data[i][3]))
                tmp.append(float(data[i][4]))
                tmp.append(float(data[i][5]))
                tmp.append(float(data[i][6]))
                tmp.append(float(data[i][7]))
                tmp.append(float(data[i][8]))
                tmp.append(float(data[i][9]))
                tmp.append(float(data[i][10]))

            feaV.append(tmp)
        feaV = np.array(feaV)

        np.save('../MTFL/features/' + datatype + '_feature_newclus.npy', feaV)

    orgmodel = TCDCNN()
    S1model = TCDCNN()
    S2model = TCDCNN()
    S3model = TCDCNN()
    orgstate_dict = torch.load('../models/' + datatype + '_tcdcn.pth')
    orgmodel.load_state_dict(orgstate_dict)

    S1state_dict = torch.load('../retrainmodels/' + datatype + '_tcdcn_retrain_VAE.pth')
    S1model.load_state_dict(S1state_dict)

    S2state_dict = torch.load('../retrainmodels/' + datatype + '_tcdcn_retrain_MID_new.pth')
    S2model.load_state_dict(S2state_dict)

    S3state_dict = torch.load('../retrainmodels/' + datatype + '_tcdcn_retrain_OUT_new.pth')
    S3model.load_state_dict(S3state_dict)

    if datatype == 'D1':
        orgdata = np.loadtxt('../MTFL/TCDCN_NEWDATA_OUT/training_' + datatype + '_VAE_MID_OUT_new_hol.txt',
                             dtype=str)
    elif datatype == 'D2' or datatype == 'org':
        orgdata = np.loadtxt('../MTFL/TCDCN_NEWDATA_OUT/training_' + datatype + '_VAE_MID_OUT_new.txt',
                             dtype=str)

    bb = np.loadtxt('../MTFL/TCDCN_NEWDATA_OUT/annotation_' + datatype + '_VAE_MID_OUT_new.txt',
                    dtype=str)
    lm = []
    for i in range(orgdata.shape[0]):
        ratio_x = 40 / (float(bb[i][2]) - float(bb[i][0]))
        ratio_y = 40 / (float(bb[i][3]) - float(bb[i][1]))
        l1, l2, l3, l4, l5, l6, l7, l8, l9, l10 = (float(orgdata[i][1]) - float(bb[i][0])) * ratio_x, (
                float(orgdata[i][2]) - float(bb[i][0])) * ratio_x, \
                                                  (float(orgdata[i][3]) - float(bb[i][0])) * ratio_x, (
                                                          float(orgdata[i][4]) - float(bb[i][0])) * ratio_x, (
                                                          float(orgdata[i][5]) - float(bb[i][0])) * ratio_x, (
                                                          float(orgdata[i][6]) - float(bb[i][1])) * ratio_y, \
                                                  (float(orgdata[i][7]) - float(bb[i][1])) * ratio_y, (
                                                          float(orgdata[i][8]) - float(bb[i][1])) * ratio_y, (
                                                          float(orgdata[i][9]) - float(bb[i][1])) * ratio_y, (
                                                          float(orgdata[i][10]) - float(bb[i][1])) * ratio_y
        lm.append([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10])

    lm = np.array(lm)
    lm = torch.from_numpy(lm)

    print(orgdata.shape)
    imageset = []
    indexes = []
    for i in tqdm(range(orgdata.shape[0])):
        imgpath = join('../MTFL', orgdata[i][0])
        temp = cv2.imread(imgpath)
        temp.astype(np.uint8)
        gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

        crop_img = gray[int(float(bb[i][1])):int(float(bb[i][3])), int(float(bb[i][0])):int(float(bb[i][2]))]
        if (crop_img.shape[0] < 40 or crop_img.shape[1] < 40):
            indexes.append(i)
            continue
        resized = cv2.resize(crop_img, (40, 40), interpolation=cv2.INTER_AREA)
        resized = resized.reshape(-1, 40, 40)
        imageset.append(resized)
    imageset = np.array(imageset)

    indexes = np.array(indexes)

    for index in reversed(indexes):
        orgdata = np.delete(orgdata, index, axis=0)
        bb = np.delete(bb, index, axis=0)
        lm = np.delete(lm, index, axis=0)

    orgsfm, orgloss = getOUTPUT(orgmodel, imageset, lm)
    print('org shape:', orgsfm.shape, orgloss.shape)

    GTsfm = lm.numpy().copy()
    print('GTsfm shape:', GTsfm.shape)

    S1sfm, S1loss = getOUTPUT(S1model, imageset, lm)
    print('S1 shape:', S1sfm.shape, S1loss.shape)

    S2sfm, S2loss = getOUTPUT(S2model, imageset, lm)
    print('S2 shape:', S2sfm.shape, S2loss.shape)

    S3sfm, S3loss = getOUTPUT(S3model, imageset, lm)
    print('S3 shape:', S3sfm.shape, S3loss.shape)

    suslist = S1loss + S2loss + S3loss - orgloss

    get_featureV(orgdata, orgsfm, GTsfm, S1sfm, S2sfm, S3sfm, orgloss, S1loss, S2loss, S3loss, suslist, datatype)


def getmodelout(model, X, Y):
    model.eval()
    sfout = []
    with torch.no_grad():
        out = model(X.float())
        sfout.append(out.numpy()[0])
    sfout = np.array(sfout)
    return sfout


def getransusp(orgmodel, S1model, S2model, S3model, X, Y):
    orgmodel.eval()
    S1model.eval()
    S2model.eval()
    S3model.eval()
    orglist = []
    S1list = []
    S2list = []
    S3list = []

    with torch.no_grad():
        org_out = orgmodel(X.float())
        S1_out = S1model(X.float())
        S2_out = S2model(X.float())
        S3_out = S3model(X.float())

        org_loss = orgmodel.loss([org_out],
                                 [Y.float()])
        S1_loss = S1model.loss([S1_out],
                               [Y.float()])
        S2_loss = S2model.loss([S2_out],
                               [Y.float()])
        S3_loss = S3model.loss([S3_out],
                               [Y.float()])

        orglist.append(org_loss)
        S1list.append(S1_loss)
        S2list.append(S2_loss)
        S3list.append(S3_loss)

    return orglist, S1list, S2list, S3list


def getrandomdata(datatype, i):
    X = np.zeros((1, 1, 40, 40))
    # print(X.shape)
    for i in range(40):
        for j in range(40):
            X[0, 0, i, j] = random.randint(0, 255)
    X = X.astype('uint8')
    X = torch.from_numpy(X)
    Y = np.zeros((1, 10))
    for i in range(10):
        Y[0, i] = 0 + (40 - 0) * np.random.random()
    Y = Y.astype('float64')
    Y = torch.from_numpy(Y)

    orgmodel = TCDCNN()
    S1model = TCDCNN()
    S2model = TCDCNN()
    S3model = TCDCNN()
    orgstate_dict = torch.load('../models/' + datatype + '_tcdcn.pth')
    orgmodel.load_state_dict(orgstate_dict)

    S1state_dict = torch.load('../retrainmodels/' + datatype + '_tcdcn_retrain_VAE.pth')
    S1model.load_state_dict(S1state_dict)

    S2state_dict = torch.load('../retrainmodels/' + datatype + '_tcdcn_retrain_MID_new.pth')
    S2model.load_state_dict(S2state_dict)

    S3state_dict = torch.load('../retrainmodels/' + datatype + '_tcdcn_retrain_OUT_new.pth')
    S3model.load_state_dict(S3state_dict)

    orgsfm = getmodelout(orgmodel, X, Y)
    S1sfm = getmodelout(S1model, X, Y)
    GTsf = Y.numpy()
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

    # fea.append(S1list[0] + S2list[0] + S3list[0] - orglist[0])#Susp指标，现在不需要了
    fea = np.array(fea)
    return fea


def Offline(feaVec, datatype):
    # isdt = np.array([int(x) for x in feaVec[:, 68]])
    feaVecsimple = feaVec[:, 0:57]

    NUM = 100
    NUM2 = 100

    cnt = 0
    newfea = []
    tmpfea = copy.deepcopy(feaVecsimple)
    ind = [x for x in range(tmpfea.shape[0])]
    random.shuffle(ind)
    tmpfea = tmpfea[ind]

    start = time.time()
    for i in range(tmpfea.shape[0]):
        if cnt < NUM:  # and int(tmpfea[i][-1])==0:
            newfea.append(tmpfea[i])
            cnt += 1
    for i in range(NUM2):
        newfea.append(getrandomdata(datatype, i))

    newfea = np.array(newfea)


    Y = []
    for i in range(NUM):
        Y.append(0)
    for i in range(NUM2):
        Y.append(1)

    Y = np.array(Y)
    print('start')
    lg = LogisticRegression(C=1.0, max_iter=1000)
    lg.fit(newfea, Y)
    print('finish')
    LRres = lg.predict_proba(feaVecsimple)  ####@@@@
    LRres = LRres[:, 1]


    newdata = []
    for i in range(feaVec.shape[0]):  # 57:78
        newdata.append([feaVec[i][57], feaVec[i][58], feaVec[i][59], feaVec[i][60],
                        feaVec[i][61], feaVec[i][62], feaVec[i][63], feaVec[i][64],
                        feaVec[i][65], feaVec[i][66], feaVec[i][67], feaVec[i][68],
                        float(LRres[i])])
                        # feaVec[i][69],feaVec[i][70],feaVec[i][71],feaVec[i][72],
                        # feaVec[i][73],feaVec[i][74],feaVec[i][75],feaVec[i][76],feaVec[i][77],feaVec[i][78],

    bb = np.loadtxt('../MTFL/TCDCN_NEWDATA_OUT/annotation_' + datatype + '_VAE_MID_OUT_new.txt',
                    dtype=str)
    newbb = []
    rank = []
    for i in range(feaVec.shape[0]):
        newbb.append([bb[i][0], bb[i][1], bb[i][2], bb[i][3], float(LRres[i])])

    for i in range(feaVec.shape[0]):
        rank.append(int(feaVec[i][68]))
    rank = np.array(rank)
    newbb = np.array(newbb, dtype=object)

    rank = rank[newbb[:, -1].argsort()[::-1]]
    feaVec = feaVec[newbb[:, -1].argsort()[::-1]]
    end = time.time()
    print('执行时间:', end - start)

    newbb = newbb[newbb[:, -1].argsort()[::-1]]

    newdata = np.array(newdata, dtype=object)

    newdata = newdata[newdata[:, -1].argsort()[::-1]]
    # np.save('F:/ICSEdata/RQ1data/MTFL/' + datatype + '_fea_offline_newclu.npy', feaVec)
    # np.save('F:/ICSEdata/RQ1data/MTFL/' + datatype + '_offline_newclu.npy', newdata)
    # np.save('F:/ICSEdata/RQ1data/MTFL/' + datatype + '_bb_offline_newclu.npy', newbb)

    print(rank)
    print("rank shape: ", rank.shape)
    save3, y, yb = RAUC(rank, bestAUC(rank))
    f = EXAM_F(rank)
    l = EXAM_L(rank)
    avg = EXAM_AVG(rank)
    save1 = PoBL(rank, 0.1)
    save2 = APFD(rank)
    return save1, save2, save3, f, l, avg, y, yb, newdata, newbb, feaVec


def Online_getAPFD_RAUC(fea):
    rank = fea[:, 68].astype('int')
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


def Online(feaVec, datatype):
    N = feaVec.shape[0]  # data num
    # args###################################################
    check_ratio = 0.25
    per_check = 200
    epochs = int((check_ratio * N) / per_check)
    logger.info('[datanum,check_ratio,per_check,epochs]=' + str([N, check_ratio, per_check, epochs]))
    ########################################################

    # offline part#######

    la = -1
    _, _, _, _, _, _, _, _, _, bb_left, fea_left = Offline(feaVec, datatype)
    logger.info('success get offline result shape:' + str(fea_left.shape))

    logger.info('start online')

    global X
    global B
    APFDlist = []
    RAUClist = []
    EPOCHlist = []

    for i in range(epochs):
        EPOCHlist.append(i)
        logger.info('run: ' + str(i + 1) + '/' + str(epochs) + ' epoch')

        fea = fea_left[0:per_check, :]
        bb = bb_left[0:per_check, :]

        fea_left = fea_left[per_check:, :]
        bb_left = bb_left[per_check:, :]

        lg = LogisticRegression(C=1.0)
        if i == 0:
            X = fea[:]
            B = bb[:]
        else:
            X = np.vstack((X, fea))
            B = np.vstack((B, bb))

        logger.info('X shape: ' + str(X.shape))
        logger.info('fea_left shape: ' + str(fea_left.shape))
        lg.fit(X[:, 0:57].astype('float32'), X[:, 68].astype('int'))

        LRres = lg.predict_proba(fea_left[:, 0:57].astype('float32'))  # 预测剩余数据
        total_res = lg.predict_proba(feaVec[:, 0:57].astype('float32'))  # 预测整个数据集

        LRres = LRres[:, 1]
        total_res = total_res[:, 1]

        fea_left = fea_left[LRres.argsort()[::-1]]  # 根据预测结果排序*剩余*数据
        bb_left = bb_left[LRres.argsort()[::-1]]
        feaVec = feaVec[total_res.argsort()[::-1]]  # 根据预测结果排序*整个*数据集

        apfd, rauc = Online_getAPFD_RAUC(feaVec)
        APFDlist.append(apfd)
        RAUClist.append(rauc)

        if i == epochs - 1:
            X = np.vstack((X, fea_left))
            B = np.vstack((B, bb_left))

    X = np.array(X)
    B = np.array(B)
    logger.info('final X shape: ' + str(X.shape))
    logger.info('final B shape: ' + str(B.shape))

    rank = X[:, 68].astype('int')
    save3, y, yb = RAUC(rank, bestAUC(rank))
    f = EXAM_F(rank)
    l = EXAM_L(rank)
    avg = EXAM_AVG(rank)
    save1 = PoBL(rank, 0.1)
    save2 = APFD(rank)
    newdata = []
    for i in range(X.shape[0]):  # 57:78
        newdata.append([X[i][57], X[i][58], X[i][59], X[i][60],
                        X[i][61], X[i][62], X[i][63], X[i][64],
                        X[i][65], X[i][66], X[i][67], X[i][68],
                        X[i][69], X[i][70], X[i][71], X[i][72],
                        X[i][73], X[i][74], X[i][75], X[i][76], X[i][77], X[i][78],
                        ])
    newdata = np.array(newdata, dtype=object)
    np.save('F:\\ICSEdata\\online_new\\MTFL\\' + datatype + '_' + '_online.npy', newdata)
    np.save('F:\\ICSEdata\\online_new\\MTFL\\' + datatype + '_' + 'bb_online.npy', B)

    y = np.array(y)
    EPOCHlist = np.array(EPOCHlist)
    APFDlist = np.array(APFDlist)
    RAUClist = np.array(RAUClist)

    np.save('F:\\ICSEdata\\online_new\\MTFL\\' + datatype + '_' + '_online_Cord.npy', y)
    np.save('F:\\ICSEdata\\online_new\\MTFL\\' + datatype + '_' + '_EPOCH_Cord.npy', EPOCHlist)
    np.save('F:\\ICSEdata\\online_new\\MTFL\\' + datatype + '_' + '_APFD_Cord.npy', APFDlist)
    np.save('F:\\ICSEdata\\online_new\\MTFL\\' + datatype + '_' + '_RAUC_Cord.npy', RAUClist)

    X = [x for x in range(101)]

    # plt.figure()
    # plt.title(datatype + ' ')
    # plt.plot(X, yb, color='blue', label='Theory Best')
    # plt.plot(X, y, color='red', label='Online')
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # plt.title(datatype + ' ')
    # plt.plot(EPOCHlist, APFDlist, color='Aqua', label='APFD')
    # plt.plot(EPOCHlist, RAUClist, color='red', label='RAUC')
    # plt.legend()
    # plt.show()
    return save1, save2, save3, f, l, avg, y




if __name__ == "__main__":
    '''
       Tips:

       You can get the features in the previous step by modifying the code(mnist_Outlier,mnist_Activation,mnist_PreLoss,mnist_mutation) 
       , which helps to reduce the algorithm complexity!

       The features are reextracted here only to demonstrate the integrity of the algorithm
    '''

    FeatureExtraction('D1')
    feaVec = np.load('../MTFL/features/D1_feature_newclus.npy',
                     allow_pickle=True)
    Offline(feaVec, 'D1')  # offline

    Online(feaVec, 'D1')  # online


