import copy
import logging
import pickle
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
from torchtext.data.utils import get_tokenizer

from train_model.models import LSTM, BiLSTM

vocab = pickle.load(open("./data/AgNews/vocab.pkl", "rb"))
device='cuda'
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


def ORGLABEL(datatype, modeltype, newdatasavepath):

    if datatype=='alldirty' or datatype=='randirty':
        nowdatapath = './data/AgNews/AgNews_NEWDATA_Out/' + datatype + 'traindata_VAE_Kmeans_LOSS_' + modeltype + '.npy'
        nowdata = np.load(nowdatapath, allow_pickle=True)
        newdata=[]
        for i in range(nowdata.shape[0]):
            newdata.append([nowdata[i][0], nowdata[i][1], nowdata[i][2], nowdata[i][3], nowdata[i][4], nowdata[i][5],
                            nowdata[i][2]])
        newdata=np.array(newdata)
        print(newdata.shape)
        np.save(newdatasavepath,newdata)
        return


    def load_data(traindata):
        trainarr = np.load(traindata)
        tokenizer = get_tokenizer('basic_english')
        traind = []
        for i in range(trainarr.shape[0]):
            traind.append(
                [trainarr[i][0], vocab.transform(sentence=tokenizer(trainarr[i][1]), max_len=100), trainarr[i][2]])
        traind = np.array(traind, dtype=object)
        return traind

    nowdatapath = './data/AgNews/AgNews_NEWDATA_Out/' + datatype + 'traindata_VAE_Kmeans_LOSS_' + modeltype + '.npy'
    lastdatapath = './data/AgNews/' + datatype + 'traindata.npy'
    orgdatapath = './data/AgNews/orgtraindata.npy'

    nowdata = np.load(nowdatapath, allow_pickle=True)
    tmpdata = load_data(lastdatapath)
    orgdata = load_data(orgdatapath)
    print(nowdata.shape)
    cnt = 0
    sum = 0
    pcnt = 0
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
                if nowdata[i][1] == lastdata[j][1]:
                    newdata.append(
                        [nowdata[i][0], nowdata[i][1], nowdata[i][2], nowdata[i][3], nowdata[i][4], nowdata[i][5],
                         lastdata[j][3]])
                    break
        else:
            newdata.append([nowdata[i][0], nowdata[i][1], nowdata[i][2], nowdata[i][3], nowdata[i][4], nowdata[i][5],
                            nowdata[i][0]])
    newdata = np.array(newdata)
    print(newdata.shape)
    np.save(newdatasavepath, newdata)

    # for i in range(newdata.shape[0]):
    #     if int(newdata[i][0])!=int(newdata[i][3]):
    #         cnt+=1
    # print(newdata.shape)
    # print(cnt)
    # for i in tqdm(range(tmpdata.shape[0])):
    #     if int(tmpdata[i][2]) == 1:
    #         for j in range(newdata.shape[0]):
    #             if orgdata[i][1]==newdata[j][1] and orgdata[i][0]==newdata[j][6]:
    #                 cnt+=1
    # print(cnt)

def FeatureExtraction(modeltype, datatype):
    def getOUTPUT(model, datapath):
        predata = np.load(datapath, allow_pickle=True)
        x_train = torch.from_numpy(np.array([x for x in predata[:, 1]]))
        y_train = torch.from_numpy(np.array([int(x) for x in predata[:, 0]]))

        loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
        model.to(device)
        model.eval()
        sfout = []
        softmax_func = nn.Softmax(dim=1)
        losslst = []
        with torch.no_grad():
            for i in tqdm(range(x_train.shape[0])):
                X = x_train[i].reshape(1, -1)
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
            tmp = [0, 0, 0, 0]
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
            for j in range(4):
                tmp.append(orgsfm[i][j])
            for j in range(4):
                tmp.append(GTsfm[i][j])
            for j in range(4):
                tmp.append(S1sfm[i][j])
            for j in range(4):
                tmp.append(S2sfm[i][j])
            for j in range(4):
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
        # np.save('F:/ICSEdata/RQ1usedata/AgNewsfeature/' + datatype + modeltype + '_feature'  '.npy', feaV)

    path = './retrainmodel/'
    if modeltype == 'LSTM':
        orgmodel = LSTM(voc_len=len(vocab), PAD=vocab.PAD)
        S1model = LSTM(voc_len=len(vocab), PAD=vocab.PAD)
        S2model = LSTM(voc_len=len(vocab), PAD=vocab.PAD)
        S3model = LSTM(voc_len=len(vocab), PAD=vocab.PAD)

        orgstate_dict = torch.load('./models/agnews_' + datatype + '_LSTM.pth')
        orgmodel.load_state_dict(orgstate_dict)

        S1state_dict = torch.load(path + 'agnews_' + datatype + '_LSTM_retrain_VAE.pth')
        S1model.load_state_dict(S1state_dict)

        S2state_dict = torch.load('./retrainmodel/agnews_' + datatype + '_LSTM_retrain_Kmeans.pth')
        S2model.load_state_dict(S2state_dict)

        S3state_dict = torch.load(path + 'agnews_' + datatype + '_LSTM_retrain_LOSS.pth')
        S3model.load_state_dict(S3state_dict)
    elif modeltype == 'BiLSTM':
        orgmodel = BiLSTM(voc_len=len(vocab), PAD=vocab.PAD)
        S1model = BiLSTM(voc_len=len(vocab), PAD=vocab.PAD)
        S2model = BiLSTM(voc_len=len(vocab), PAD=vocab.PAD)
        S3model = BiLSTM(voc_len=len(vocab), PAD=vocab.PAD)

        orgstate_dict = torch.load('./models/agnews_' + datatype + '_BiLSTM.pth')
        orgmodel.load_state_dict(orgstate_dict)

        S1state_dict = torch.load(path + 'agnews_' + datatype + '_BiLSTM_retrain_VAE.pth')
        S1model.load_state_dict(S1state_dict)

        S2state_dict = torch.load('./retrainmodel/agnews_' + datatype + '_BiLSTM_retrain_Kmeans.pth')
        S2model.load_state_dict(S2state_dict)

        S3state_dict = torch.load(path + 'agnews_' + datatype + '_BILSTM_retrain_LOSS.pth')
        S3model.load_state_dict(S3state_dict)


    DP = 'F:/ICSEdata/RQ1usedata/AgNews/'

    datapath = DP + datatype + 'traindata_VAE_Kmeans_LOSS_' + modeltype  + '.npy'

    predata = np.load(datapath, allow_pickle=True)
    y_train = torch.from_numpy(np.array([int(x) for x in predata[:, 0]]))
    is_dirty = torch.from_numpy(np.array([int(x) for x in predata[:, 2]]))
    orgmodel.eval()
    S1model.eval()
    S2model.eval()
    S3model.eval()

    start=time.time()

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


    get_featureV(datapath, orgsfm, GTsfm, S1sfm, S2sfm, S3sfm, orgloss, S1loss, S2loss, S3loss, datatype,
                 modeltype)

    end=time.time()
    print('执行时间:',end-start)

def getmodelout(model, X, Y):
    model.eval()
    sfout = []
    softmax_func = nn.Softmax(dim=1)
    X = torch.from_numpy(X)
    X = X.int()

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
    X = torch.from_numpy(X)
    X = X.int()
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
    X = np.zeros((1, 100))
    # print(X.shape)
    for i in range(100):
        X[0][i] = random.randint(0, 95804)
    Y = lb

    if modeltype == 'LSTM':
        orgmodel = LSTM(voc_len=len(vocab), PAD=vocab.PAD)
        S1model = LSTM(voc_len=len(vocab), PAD=vocab.PAD)
        S2model = LSTM(voc_len=len(vocab), PAD=vocab.PAD)
        S3model = LSTM(voc_len=len(vocab), PAD=vocab.PAD)

        orgstate_dict = torch.load('./models/agnews_' + datatype + '_LSTM.pth')
        orgmodel.load_state_dict(orgstate_dict)

        S1state_dict = torch.load('./retrainmodel/agnews_' + datatype + '_LSTM_retrain_VAE.pth')
        S1model.load_state_dict(S1state_dict)

        S2state_dict = torch.load('./retrainmodel/agnews_' + datatype + '_LSTM_retrain_Kmeans.pth')
        S2model.load_state_dict(S2state_dict)

        S3state_dict = torch.load('./retrainmodel/agnews_' + datatype + '_LSTM_retrain_Confident.pth')
        S3model.load_state_dict(S3state_dict)
    elif modeltype == 'BiLSTM':
        orgmodel = BiLSTM(voc_len=len(vocab), PAD=vocab.PAD)
        S1model = BiLSTM(voc_len=len(vocab), PAD=vocab.PAD)
        S2model = BiLSTM(voc_len=len(vocab), PAD=vocab.PAD)
        S3model = BiLSTM(voc_len=len(vocab), PAD=vocab.PAD)

        orgstate_dict = torch.load('./models/agnews_' + datatype + '_BiLSTM.pth')
        orgmodel.load_state_dict(orgstate_dict)

        S1state_dict = torch.load('./retrainmodel/agnews_' + datatype + '_BiLSTM_retrain_VAE.pth')
        S1model.load_state_dict(S1state_dict)

        S2state_dict = torch.load('./retrainmodel/agnews_' + datatype + '_BiLSTM_retrain_Kmeans.pth')
        S2model.load_state_dict(S2state_dict)

        S3state_dict = torch.load('./retrainmodel/agnews_' + datatype + '_BILSTM_retrain_Confident.pth')
        S3model.load_state_dict(S3state_dict)

    orgsfm = getmodelout(orgmodel, X, Y)
    S1sfm = getmodelout(S1model, X, Y)
    GTsf = []

    tmp = [0, 0, 0, 0]
    tmp[int(Y)] = 1
    GTsf.append(tmp)
    GTsf = np.array(GTsf)
    S2sfm = getmodelout(S2model, X, Y)
    S3sfm = getmodelout(S3model, X, Y)
    orglist, S1list, S2list, S3list = getransusp(orgmodel, S1model, S2model, S3model, X, Y)

    fea = []
    for i in range(4):
        fea.append(orgsfm[0][i])
    for i in range(4):
        fea.append(GTsf[0][i])
    for i in range(4):
        fea.append(S1sfm[0][i])
    for i in range(4):
        fea.append(S2sfm[0][i])
    for i in range(4):
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
    NUM2 = 4

    cnt = 0
    newfea = []
    tmpfea = copy.deepcopy(feaVecsimple)
    ind = [x for x in range(tmpfea.shape[0])]
    random.shuffle(ind)
    tmpfea = tmpfea[ind]
    # for i in range(tmpfea.shape[0]):
    #     tmpfea[i], tmpfea[ind[i]] = tmpfea[ind[i]], tmpfea[i]
    start=time.time()
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
    newdata = np.array(newdata)
    end=time.time()
    print('执行时间:',end-start)


    feaVec = feaVec[newdata[:, 4].argsort()[::-1]]


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
    start1=time.time()
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
    np.save('F:\\ICSEdata\\online_new\\AgNews\\' + datatype + '_' + modeltype + '_online.npy', newdata)

    y = np.array(y)
    EPOCHlist = np.array(EPOCHlist)
    APFDlist = np.array(APFDlist)
    RAUClist = np.array(RAUClist)

    np.save('F:\\ICSEdata\\online_new\\AgNews\\' + datatype + '_' + modeltype + '_online_Cord.npy', y)
    np.save('F:\\ICSEdata\\online_new\\AgNews\\' + datatype + '_' + modeltype + '_EPOCH_Cord.npy', EPOCHlist)
    np.save('F:\\ICSEdata\\online_new\\AgNews\\' + datatype + '_' + modeltype + '_APFD_Cord.npy', APFDlist)
    np.save('F:\\ICSEdata\\online_new\\AgNews\\' + datatype + '_' + modeltype + '_RAUC_Cord.npy', RAUClist)

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

from CL import CLEANLAB_CV
def CleanLab(datatype,modeltype):
    CLEANLAB_CV(datatype,modeltype)

from torch.utils.data import TensorDataset
def DeepState(feaVec,modeltype,datatype):
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    layer = 'lstm'
    activation = {}
    if modeltype=='LSTM':
        model = LSTM(voc_len=len(vocab), PAD=vocab.PAD)
        orgstate_dict = torch.load('./models/agnews_' + datatype + '_LSTM.pth')
        model.load_state_dict(orgstate_dict)
    elif modeltype == 'BiLSTM':
        model = BiLSTM(voc_len=len(vocab), PAD=vocab.PAD)
        orgstate_dict = torch.load('./models/agnews_' + datatype + '_BiLSTM.pth')
        model.load_state_dict(orgstate_dict)

    model.lstm.register_forward_hook(get_activation(layer))
    x_data = torch.from_numpy(np.array([x for x in feaVec[:, -3]]))
    y_data = torch.from_numpy(np.array([int(x) for x in feaVec[:, -4]]))
    isdirty = torch.from_numpy(np.array([int(x) for x in feaVec[:, -2]]))
    traindataset = TensorDataset(x_data, isdirty)
    train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=16)
    cnt = 0
    print(x_data.shape)
    s = 0
    device = "cuda"
    model.to(device)
    state = []
    with torch.no_grad():
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            # y=y.to(device)
            # x = model(x)
            # _, prediction = torch.max(x, 1)
            # s+=torch.sum(prediction == y)
            t = model.embedding(x)
            output, (h_n, c_n) = model.lstm(t)
            # print(output.shape)
            label_seq = []
            for j in range(100):
                out = model.fc1(output[:, j, :])
                out = model.relu(out)
                out = model.fc2(out)
                _, prediction = torch.max(out, 1)
                prediction = prediction.to("cpu")
                label_seq.append(prediction.numpy())
                # label_seq.append([prediction])
            # label_seq=np.array(label_seq)
            label_seq = np.array(label_seq)
            m = label_seq.shape[1]

            for ind in range(m):
                s1 = set()
                for j in range(100 - 1):
                    s1.add(str(label_seq[j][ind]) + str(label_seq[j + 1][ind]))
                cr = 0.
                sum = 0.
                for j in range(100 - 1):
                    if label_seq[j][ind] != label_seq[j + 1][ind]:
                        cr += ((j + 1) ** 2)
                    sum += ((j + 1) ** 2)
                state.append([int(y[ind]), s1, cr / sum, cnt, 0])
                cnt += 1
    state = np.array(state, dtype=object)
    rank = []
    for i in range(state.shape[0]):
        rank.append([feaVec[i][-2], float(state[i][2])])
    rank = np.array(rank)
    newdata = []

    for i in range(feaVec.shape[0]):
        newdata.append([feaVec[i][-4], feaVec[i][-3], feaVec[i][-2], feaVec[i][-1], float(state[i][2])])
    newdata = np.array(newdata)

    newdata = newdata[newdata[:, 4].argsort()[::-1]]
    rank = rank[rank[:, 1].argsort()[::-1]]

    # save3, y, yb = RAUC(rank, bestAUC(rank))
    # f = EXAM_F(rank)
    # l = EXAM_L(rank)
    # avg = EXAM_AVG(rank)
    # save1 = PoBL(rank, 0.1)
    # save2 = APFD(rank)
    # return save1, save2, save3, f, l, avg, y, yb, newdata
    # print(s / 120000)


def RQ1(modeltype, datatype, excelx, sheet):
    plt.figure()
    Coordinatedata = []  # X,theory,offline,online

    # OffLine
    bestt1 = 0
    bestt2 = 0
    bestf = 0
    bestl = 0
    bestavg = 0
    bestscore = 0
    besty = 0
    theory = 0
    savedata = 0
    random.seed(6657)
    feaVec = np.load('F:/ICSEdata/RQ1usedata/AgNewsfeature/' + datatype + modeltype + '_feature.npy',
                     allow_pickle=True)  # LeNet5
    for _ in range(10):
        print(_)
        t1, t2, t3, f, l, avg, y, yb, data = AT(feaVec, modeltype, datatype)
        if t3 > bestscore:
            bestt1 = t1
            bestt2 = t2
            bestf = f
            bestl = l
            bestavg = avg
            bestscore = t3
            besty = y
            theory = yb
            savedata = data
    writexcel(sheet, excelx, 0, bestf, '', 0)
    writexcel(sheet, excelx, 0, bestl, '', 6)
    writexcel(sheet, excelx, 0, bestavg, '', 12)
    writexcel(sheet, excelx + 29, 0, bestt1, '', 0)
    writexcel(sheet, excelx + 29, 0, bestt2, '', 6)
    writexcel(sheet, excelx + 29, 0, bestscore, '', 12)
    np.save('F:/ICSEdata/RQ1data/AgNews/' + datatype + modeltype + '_offline.npy', savedata)
    X = [x for x in range(101)]
    plt.plot(X, theory, color='blue', label='Theory Best')
    plt.plot(X, besty, color='pink', label='Offline')
    Coordinatedata.append(X)
    Coordinatedata.append(theory)
    Coordinatedata.append(besty)

    # Online
    random.seed(6657)
    t1, t2, t3, f, l, avg, y, yb, onlinedata = UPDATE(feaVec, modeltype, datatype)
    writexcel(sheet, excelx, 1, f, '', 0)
    writexcel(sheet, excelx, 1, l, '', 6)
    writexcel(sheet, excelx, 1, avg, '', 12)
    writexcel(sheet, excelx + 29, 1, t1, '', 0)
    writexcel(sheet, excelx + 29, 1, t2, '', 6)
    writexcel(sheet, excelx + 29, 1, t3, '', 12)
    plt.plot(X, y, color='red', label='Online')
    np.save('F:/ICSEdata/RQ1data/AgNews/' + datatype + modeltype + '_online.npy', onlinedata)
    Coordinatedata.append(y)

    # Random
    y = [x for x in range(101)]
    plt.plot(X, y, color='black', label='Random')

    # DeepGini
    t1, t2, t3, f, l, avg, y, yb, ginidata = DeepGini(feaVec, 4)
    np.save('F:/ICSEdata/RQ1data/AgNews/' + datatype + modeltype + '_gini.npy', ginidata)
    writexcel(sheet, excelx, 2, f, '', 0)
    writexcel(sheet, excelx, 2, l, '', 6)
    writexcel(sheet, excelx, 2, avg, '', 12)
    writexcel(sheet, excelx + 29, 2, t1, '', 0)
    writexcel(sheet, excelx + 29, 2, t2, '', 6)
    writexcel(sheet, excelx + 29, 2, t3, '', 12)
    Coordinatedata.append(y)
    plt.plot(X, y, linestyle='--', color='#8A2BE2', label='DeepGini')


    #DeepState
    t1, t2, t3, f, l, avg, y, yb, statedata = DeepState(feaVec, modeltype,datatype)
    np.save('F:/ICSEdata/RQ1data/AgNews/' + datatype + modeltype + '_state.npy', ginidata)
    writexcel(sheet, excelx, 3, f, '', 0)
    writexcel(sheet, excelx, 3, l, '', 6)
    writexcel(sheet, excelx, 3, avg, '', 12)
    writexcel(sheet, excelx + 29, 3, t1, '', 0)
    writexcel(sheet, excelx + 29, 3, t2, '', 6)
    writexcel(sheet, excelx + 29, 3, t3, '', 12)
    Coordinatedata.append(y)
    plt.plot(X, y, linestyle='--', color='#A52A2A', label='DeepState')



    # cleanlab
    random.seed(6657)
    t1, t2, t3, f, l, avg, y, yb = CleanLab(datatype,modeltype)
    # np.save('F:/ICSEdata/RQ1data/AgNews/' + datatype + modeltype + '_cleanlab.npy', ginidata)
    writexcel(sheet, excelx, 5, f, '', 0)
    writexcel(sheet, excelx, 5, l, '', 6)
    writexcel(sheet, excelx, 5, avg, '', 12)
    writexcel(sheet, excelx + 29, 5, t1, '', 0)
    writexcel(sheet, excelx + 29, 5, t2, '', 6)
    writexcel(sheet, excelx + 29, 5, t3, '', 12)
    Coordinatedata.append(y)
    plt.plot(X, y, linestyle='--', color='Aqua', label='CleanLab')



    Coordinatedata = np.array(Coordinatedata)
    np.save('F:/ICSEdata/RQ1data/AgNews/' + datatype + modeltype + '_Coordinatedata.npy', Coordinatedata)

    plt.xlabel(r'$Percentage\ of\ test\ case\ executed$')
    plt.ylabel(r'$Percentage\ of\ fault\ detected$')
    plt.legend()
    if datatype == 'alllabel':
        plt.savefig('F:/ICSEdata/RQ1picture/AgNews/' + 'RandomLabelNoise_' + modeltype + '.jpg', dpi=1200)  # 指定分辨率保存
    elif datatype == 'ranlabel':
        plt.savefig('F:/ICSEdata/RQ1picture/AgNews/' + 'SepcificLabelNoise_' + modeltype + '.jpg', dpi=1200)  # 指定分辨率保存
    elif datatype == 'alldirty':
        plt.savefig('F:/ICSEdata/RQ1picture/AgNews/' + 'RandomDataNoise_' + modeltype + '.jpg', dpi=1200)  # 指定分辨率保存
    elif datatype == 'randirty':
        plt.savefig('F:/ICSEdata/RQ1picture/AgNews/' + 'SpecificDataNoise_' + modeltype + '.jpg', dpi=1200)  # 指定分辨率保存
    return sheet

def NEW_ONLINE(datatype, modeltype, excelx, sheet):
    feaVec = np.load('F:/ICSEdata/RQ1usedata/AgNewsfeature/' + datatype + modeltype + '_feature.npy',
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
    feaVec = np.load('F:/ICSEdata/RQ1usedata/AgNewsfeature/alllabelBiLSTM_feature.npy',
                     allow_pickle=True)
    start=time.time()
    DeepState(feaVec,'BiLSTM','alllabel')
    end=time.time()
    print('执行时间:',end-start)
    # FeatureExtraction('BiLSTM', 'alllabel')
    # dtlist = ['alllabel', 'ranlabel', 'alldirty', 'randirty']
    # mdlist = ['LSTM', 'BiLSTM']
    # workbook = xlwt.Workbook()  # 新建一个工作簿
    # sheet = workbook.add_sheet('1')  # 在工作簿中新建一个表格
    # row = -1
    # for MODELTYPE in mdlist:
    #     for DATATYPE in dtlist:
    #         row += 1
    #         logger.info('now run:' + MODELTYPE + ' ' + DATATYPE)
    #         sheet = NEW_ONLINE(DATATYPE, MODELTYPE, row, sheet)
    #         workbook.save('F:\\ICSEdata\\online_new\\AgNews\\AgNews_RQ1.xls')
    # dtlist = ['alllabel', 'ranlabel', 'alldirty', 'randirty']
    # datasetname = 'AgNews'
    # mdlist = ['LSTM', 'BiLSTM']
    #
    # workbook = xlwt.Workbook()  # 新建一个工作簿
    # sheet = workbook.add_sheet('1')  # 在工作簿中新建一个表格
    # row = -1
    # for MODELTYPE in mdlist:
    #     for DATATYPE in dtlist:
    #         row += 1
    #         sheet = RQ1(MODELTYPE, DATATYPE, row, sheet)
    # workbook.save('F:/ICSEdata/excel/' + datasetname + '_RQ1.xls')

    # ORGLABEL('randirty', 'BiLSTM', 'F:/ICSEdata/RQ1usedata/AgNews/randirtytraindata_VAE_Kmeans_LOSS_BiLSTM.npy')
