import copy
import pickle
import random

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
from train_model.models import TCDCNN
from os.path import isfile, join


def PoBL(ranklist, ratio):
    n = ranklist.shape[0]
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
    for i in range(ranklist.shape[0]):
        if ranklist[i][0] == 1:
            bestlist.append(ranklist[i])
    for i in range(ranklist.shape[0]):
        if ranklist[i][0] == 0:
            bestlist.append(ranklist[i])
    bestlist = np.array(bestlist)
    return bestlist


def EXAM_F(ranklist):
    n = ranklist.shape[0]
    pos = -1
    for i in range(n):
        if ranklist[i][0] == 1:
            pos = i
            break
    return (pos + 1) / n
    # print('EXAM_F score: ', (pos + 1) / n)


def EXAM_L(ranklist):
    n = ranklist.shape[0]
    pos = -1
    for i in range(n - 1, -1, -1):
        if ranklist[i][0] == 1:
            pos = i
            break
    return (pos + 1) / n
    # print('EXAM_L score: ', (pos + 1) / n)


def EXAM_AVG(ranklist):
    n = ranklist.shape[0]
    m = 0
    tf = 0
    for i in range(n):
        if ranklist[i][0] == 1:
            tf += i
            m += 1
    return tf / (n * m)
    # print('EXAM_AVG score: ', tf / (n * m))


def APFD(ranklist):
    n = ranklist.shape[0]
    m = 0
    tf = 0
    for i in range(n):
        if ranklist[i][0] == 1:
            tf += i
            m += 1

    # print('APFD score: ', 1 - (tf / (n * m)) + (1 / (2 * n)))
    return 1 - (tf / (n * m)) + (1 / (2 * n))


def getlockmodelTCDCNN(modelpath):
    premodel = TCDCNN()

    state_dict = torch.load(modelpath)
    premodel.load_state_dict(state_dict)
    for param in premodel.parameters():
        param.requires_grad = False

    premodel.linear_1 = nn.Linear(256, 10)

    return premodel


def getOUTPUT(model, imageset, lm):
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


def get_featureV(data, orgsfm, GTsfm, S1sfm, S2sfm, S3sfm, orgloss, S1loss, S2loss, S3loss, suslist, datatype,dataratio):
    is_dirty = torch.from_numpy(np.array([int(x) for x in data[:, 15]]))
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
        tmp.append(int(data[i][16]))
        tmp.append(int(data[i][17]))
        tmp.append(int(data[i][18]))
        tmp.append(orgloss[i])
        tmp.append(S1loss[i])
        tmp.append(S2loss[i])
        tmp.append(S3loss[i])
        tmp.append(suslist[i])
        tmp.append(int(is_dirty[i]))
        feaV.append(tmp)

    feaV = np.array(feaV)
    np.save('../MTFL/features/randomweight_' + datatype + '_feature'+dataratio+'.npy', feaV)


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


def getrandomdata(datatype, i, ablation):
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


def AT(feaVec, datatype, is_dirty, ablation):
    NUM = 100
    NUM2 = 100

    cnt = 0
    newfea = []
    tmpfea = copy.deepcopy(feaVec)
    ind = [x for x in range(tmpfea.shape[0])]
    random.shuffle(ind)
    tmpfea = tmpfea[ind]
    # for i in range(tmpfea.shape[0]):
    #     tmpfea[i], tmpfea[ind[i]] = tmpfea[ind[i]], tmpfea[i]

    for i in range(tmpfea.shape[0]):
        if cnt < NUM:  # and int(tmpfea[i][-1])==0:
            newfea.append(tmpfea[i][0:-2])
            cnt += 1
    for i in range(NUM2):
        newfea.append(getrandomdata(datatype, i, ablation))

    newfea = np.array(newfea)
    # print(newfea.shape)

    Y = []
    for i in range(NUM):
        Y.append(0)
    for i in range(NUM2):
        Y.append(1)

    Y = np.array(Y)
    lg = LogisticRegression(C=1.0, max_iter=1000)
    lg.fit(newfea, Y)

    LRres = lg.predict_proba(feaVec[0:feaVec.shape[0], 0:-2])  ####@@@@
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


def UPDATE(feaVec, datatype, is_dirty):
    check_ratio = 0.25
    per_check = 50
    FIRST_NUM = 100

    check_num = int((0.25 - FIRST_NUM / feaVec.shape[0]) * (feaVec.shape[0] - FIRST_NUM))

    # 打乱
    ind = [x for x in range(feaVec.shape[0])]
    random.shuffle(ind)
    feaVec = feaVec[ind]
    # for i in range(feaVec.shape[0]):
    #     feaVec[i], feaVec[ind[i]] = feaVec[ind[i]], feaVec[i]

    # 第一次选择
    fea_start = feaVec[0:FIRST_NUM, :-2]
    lb_start = feaVec[0:FIRST_NUM, -1]
    print(lb_start.shape)
    if lb_start.sum() < 1:
        newfea = []
        newlb = []
        # for i in range(10):
        newfea.append(getrandomdata(datatype, i))
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
        fea = fea_left[FIRST_NUM + i * per_check:FIRST_NUM + (i + 1) * per_check, :]
        lb = lb_left[FIRST_NUM + i * per_check:FIRST_NUM + (i + 1) * per_check]

        fea_left = fea_left[per_check:, :]
        lb_left = lb_left[per_check:]

        lg = LogisticRegression(C=1.0)
        X = np.vstack((X, fea))
        Y = np.concatenate((Y, lb))
        # print('X Y shape',X.shape,Y.shape)
        # print('fea_left shape',fea_left.shape)
        lg.fit(X, Y)

        LRres = lg.predict_proba(fea_left)  ####@@@@
        LRres = LRres[:, 1]

        fea_left = fea_left[LRres.argsort()[::-1]]
        lb_left = lb_left[LRres.argsort()[::-1]]

    ##结果统计
    LRres = lg.predict_proba(feaVec[:, :-2])  ####@@@@
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


def PROCESS(datatype, ablation, pattern):
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

    orgdata = np.loadtxt('../MTFL/TCDCN_NEWDATA_OUT/training_' + datatype + '_VAE_MID_OUT_new.txt', dtype=str)
    if pattern=='getfeature':
        bb = np.loadtxt('../MTFL/TCDCN_NEWDATA_OUT/annotation_' + datatype + '_VAE_MIDVAE_OUT'+ dataratio +'.txt', dtype=str)
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

        orgsfm, orgloss= getOUTPUT(orgmodel, imageset, lm)
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

        get_featureV(orgdata, orgsfm, GTsfm, S1sfm, S2sfm, S3sfm, orgloss, S1loss, S2loss, S3loss, suslist, datatype )
    elif pattern=='AT':
        feaVec = np.load('../MTFL/features/' + datatype + '_feature_newclus.npy')
        stay = [x for x in range(57)]
        stay.append(66)
        stay.append(68)
        mv = []
        for i in range(79):
            if i not in stay:
                mv.append(i)
        feaVec = np.delete(feaVec, mv, axis=1)
        is_dirty = torch.from_numpy(np.array([int(x) for x in orgdata[:, 15]]))

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

        return AT(feaVec, datatype, is_dirty, ablation)
        # UPDATE(feaVec, datatype, is_dirty)
        return

        get_susp_rank_res(feaVec[:, -2], is_dirty)

        num = 100

        feaVec = feaVec[feaVec[:, -2].argsort()[::-1]]

        X = np.vstack((feaVec[0:num, 0:-1], feaVec[-num:orgdata.shape[0], 0:-1]))  ####@@@@

        Y = np.concatenate((feaVec[0:num, -1], feaVec[-num:orgdata.shape[0], -1]))

        print(X.shape, Y.shape)

        lg = LogisticRegression(C=1.0)
        lg.fit(X, Y)

        LRres = lg.predict_proba(feaVec[0:orgdata.shape[0], 0:-1])  ####@@@@
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
    dtlist = ['D1', 'D2']

    # args:
    MODELTYPE = 'TCDCNN'

    row = -1
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet('1')  # 在工作簿中新建一个表格
    # for i in range(PALL):

    for DATATYPE in dtlist:
        row += 1
        runlist = ['all', 'input', 'hidden', 'output', 'input+hidden', 'hidden+output', 'input+output']
        # runlist = ['_']
        PoBL_all = []
        APFD_all = []
        RAUC_all = []
        for ab in runlist:
            save1 = 0
            save2 = 0
            save3 = 0
            savef = 0
            savel = 0
            saveavg = 0

            PoBL_spc = []
            APFD_spc = []
            RAUC_spc = []
            random.seed(6657)
            for _ in range(PALL):
                print('now run:' + ab + ' ' + str(_) + '/' + str(PALL))
                t1, t2, t3, f, l, avg = PROCESS(datatype=DATATYPE, ablation=ab,pattern='AT')
                print(t1)
                if ab == 'all':
                    PoBL_all.append(t1)
                    APFD_all.append(t2)
                    RAUC_all.append(t3)
                else:
                    PoBL_spc.append(t1)
                    APFD_spc.append(t2)
                    RAUC_spc.append(t3)

                save1 += t1
                save2 += t2
                save3 += t3
                savef += f
                savel += l
                saveavg += avg

            save1 /= PALL
            save2 /= PALL
            save3 /= PALL

            savef /= PALL
            savel /= PALL
            saveavg /= PALL

            save1 = format(save1, '.4f')
            save2 = format(save2, '.4f')
            save3 = format(save3, '.4f')

            savef = format(savef, '.4f')
            savel = format(savel, '.4f')
            saveavg = format(saveavg, '.4f')

            PoBL_WTL = ''
            APFD_WTL = ''
            RAUC_WTL = ''
            if ab != 'all':
                PoBL_WTL = check(PoBL_all, PoBL_spc)
                APFD_WTL = check(APFD_all, APFD_spc)
                RAUC_WTL = check(RAUC_all, RAUC_spc)

            file = open('save.txt', 'a')
            s = MODELTYPE + ' ' + DATATYPE + ' ' + ab + ': ' + str(save1) + ' ' + str(save2) + ' ' + str(save3) \
                + ' ' + str(savef) + ' ' + str(savel) + ' ' + str(
                saveavg) + ' ' + PoBL_WTL + ' ' + APFD_WTL + ' ' + RAUC_WTL + '\n'
            file.write(s)
            file.close()

            writexcel(sheet, row, ab, save1, PoBL_WTL, 0)
            writexcel(sheet, row, ab, save2, APFD_WTL, 8)
            writexcel(sheet, row, ab, save3, RAUC_WTL, 16)

    workbook.save('C:/Users/WSHdeWindows/Desktop/res_newclus.xls')


PALL = 10


def PROCESS_randomweight(datatype, ablation):
    orgmodel = TCDCNN()
    S1model = getlockmodelTCDCNN('../models/' + datatype + '_tcdcn.pth')
    S2model = getlockmodelTCDCNN('../models/' + datatype + '_tcdcn.pth')
    S3model = getlockmodelTCDCNN('../models/' + datatype + '_tcdcn.pth')

    orgstate_dict = torch.load('../models/' + datatype + '_tcdcn.pth')
    orgmodel.load_state_dict(orgstate_dict)

    S1state_dict = torch.load('../retrainmodels/' + datatype + '_tcdcn_retrain_VAE_randomweight.pth')
    S1model.load_state_dict(S1state_dict)

    S2state_dict = torch.load('../retrainmodels/' + datatype + '_tcdcn_retrain_MIDVAE_randomweight.pth')
    S2model.load_state_dict(S2state_dict)

    S3state_dict = torch.load('../retrainmodels/' + datatype + '_tcdcn_retrain_OUTLOS_randomweight.pth')
    S3model.load_state_dict(S3state_dict)

    orgdata = np.loadtxt('../MTFL/TCDCN_NEWDATA_OUT/training_' + datatype + '_VAE_MIDVAE_OUT.txt', dtype=str)
    # bb = np.loadtxt('../MTFL/TCDCN_NEWDATA_OUT/annotation_' + datatype + '_VAE_MIDVAE_OUT.txt', dtype=str)
    # lm = []
    # for i in range(orgdata.shape[0]):
    #     ratio_x = 40 / (float(bb[i][2]) - float(bb[i][0]))
    #     ratio_y = 40 / (float(bb[i][3]) - float(bb[i][1]))
    #     l1, l2, l3, l4, l5, l6, l7, l8, l9, l10 = (float(orgdata[i][1]) - float(bb[i][0])) * ratio_x, (
    #             float(orgdata[i][2]) - float(bb[i][0])) * ratio_x, \
    #                                               (float(orgdata[i][3]) - float(bb[i][0])) * ratio_x, (
    #                                                       float(orgdata[i][4]) - float(bb[i][0])) * ratio_x, (
    #                                                       float(orgdata[i][5]) - float(bb[i][0])) * ratio_x, (
    #                                                       float(orgdata[i][6]) - float(bb[i][1])) * ratio_y, \
    #                                               (float(orgdata[i][7]) - float(bb[i][1])) * ratio_y, (
    #                                                       float(orgdata[i][8]) - float(bb[i][1])) * ratio_y, (
    #                                                       float(orgdata[i][9]) - float(bb[i][1])) * ratio_y, (
    #                                                       float(orgdata[i][10]) - float(bb[i][1])) * ratio_y
    #     lm.append([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10])
    #
    # lm = np.array(lm)
    # lm = torch.from_numpy(lm)
    #
    # print(orgdata.shape)
    # imageset = []
    # indexes = []
    # for i in tqdm(range(orgdata.shape[0])):
    #     imgpath = join('../MTFL', orgdata[i][0])
    #     temp = cv2.imread(imgpath)
    #     temp.astype(np.uint8)
    #     gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    #
    #     crop_img = gray[int(float(bb[i][1])):int(float(bb[i][3])), int(float(bb[i][0])):int(float(bb[i][2]))]
    #     if (crop_img.shape[0] < 40 or crop_img.shape[1] < 40):
    #         indexes.append(i)
    #         continue
    #     resized = cv2.resize(crop_img, (40, 40), interpolation=cv2.INTER_AREA)
    #     resized = resized.reshape(-1, 40, 40)
    #     imageset.append(resized)
    # imageset = np.array(imageset)
    #
    # indexes = np.array(indexes)
    #
    # for index in reversed(indexes):
    #     orgdata = np.delete(orgdata, index, axis=0)
    #     bb = np.delete(bb, index, axis=0)
    #     lm = np.delete(lm, index, axis=0)
    #
    # orgsfm, orgloss= getOUTPUT(orgmodel, imageset, lm)
    # print('org shape:', orgsfm.shape, orgloss.shape)
    #
    # GTsfm = lm.numpy().copy()
    # print('GTsfm shape:', GTsfm.shape)
    #
    # S1sfm, S1loss = getOUTPUT(S1model, imageset, lm)
    # print('S1 shape:', S1sfm.shape, S1loss.shape)
    #
    # S2sfm, S2loss = getOUTPUT(S2model, imageset, lm)
    # print('S2 shape:', S2sfm.shape, S2loss.shape)
    #
    # S3sfm, S3loss = getOUTPUT(S3model, imageset, lm)
    # print('S3 shape:', S3sfm.shape, S3loss.shape)
    #
    # suslist = S1loss + S2loss + S3loss - orgloss
    #
    # get_featureV(orgdata, orgsfm, GTsfm, S1sfm, S2sfm, S3sfm, orgloss, S1loss, S2loss, S3loss, suslist, datatype)
    feaVec = np.load('../MTFL/features/randomweight_' + datatype + '_feature.npy')
    is_dirty = torch.from_numpy(np.array([int(x) for x in orgdata[:, 15]]))

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

    return AT(feaVec, datatype, is_dirty, ablation)


def RQ2_2():
    dtlist = ['D1', 'D2']

    # args:
    MODELTYPE = 'TCDCNN'

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
            print('now run:' + 'direct' + ' ' + DATATYPE + ' ' + str(_) + '/' + str(PALL))
            t1, t2, t3, f, l, avg = PROCESS(datatype=DATATYPE, ablation='all')
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

        random.seed(6657)
        for _ in range(PALL):
            print('now run:' + 'randomweight' + ' ' + DATATYPE + ' ' + str(_) + '/' + str(PALL))
            t1, t2, t3, f, l, avg = PROCESS_randomweight(datatype=DATATYPE, ablation='all')
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

def RQ2_3():
    dtlist = ['D1','D2']

    # args:
    # MODELTYPE = 'VGG'
    dataratio = '020'

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
            print('now run:' + dataratio + ' ' + DATATYPE + ' ' + str(_) + '/' + str(PALL))
            t1, t2, t3, f, l, avg = PROCESS(datatype=DATATYPE, ablation='all',
                                                   dataratio=dataratio, pattern='AT')
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
            t1, t2, t3, f, l, avg = PROCESS(datatype=DATATYPE, ablation='all',
                                                   dataratio='', pattern='AT')
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
        s1 = 'TCDCNN' + ' ' + DATATYPE + ' ' + 'direct' + ': ' + str(save_DT[0]) + ' ' + str(save_DT[1]) + ' ' + str(
            save_DT[2]) \
             + ' ' + str(save_DT[3]) + ' ' + str(save_DT[4]) + ' ' + str(
            save_DT[5]) + '\n'
        s2 = 'TCDCNN' + ' ' + DATATYPE + ' ' + dataratio + ': ' + str(save_RT[0]) + ' ' + str(
            save_RT[1]) + ' ' + str(
            save_RT[2]) \
             + ' ' + str(save_RT[3]) + ' ' + str(save_RT[4]) + ' ' + str(
            save_RT[5]) + ' ' + PoBL_WTL + ' ' + APFD_WTL + ' ' + RAUC_WTL + '\n'
        file.write(s1)
        file.write(s2)
        file.close()

        # writexcel(sheet, row, 5, save_DT[0], '', 0)
        # writexcel(sheet, row, 5, save_DT[1], '', 6)
        # writexcel(sheet, row, 5, save_DT[2], '', 12)

        writexcel(sheet, row, 0, save_RT[0], PoBL_WTL, 0)
        writexcel(sheet, row, 0, save_RT[1], APFD_WTL, 6)
        writexcel(sheet, row, 0, save_RT[2], RAUC_WTL, 12)
    workbook.save('C:/Users/WSHdeWindows/Desktop/res.xls')


if __name__ == "__main__":
    RQ2_1()
    # dtlist = ['D1', 'D2']
    # # mdlist = ['ResNet', 'VGG']
    # dellist = ['003', '010', '020']
    # for datatype in dtlist:
    #     for dataratio in dellist:
    #         PROCESS(datatype=datatype,ablation='',pattern='getfeature',dataratio=dataratio)
    # getrandomdata(datatype='D1', i=1)
    # PROCESS(datatype='D2')
    # PROCESS_randomweight(datatype='D2',ablation='all')
