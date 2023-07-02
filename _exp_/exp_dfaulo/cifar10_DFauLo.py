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
        np.save('./data/CIFA10/CIFA10_PNG/' + datatype + 'traindata_VAE_Kmeans_LOSS_' + modeltype + '.npy',
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

    print(newdata.shape)
    np.save('./data/CIFA10/CIFA10_PNG/' + datatype + 'traindata_VAE_Kmeans_LOSS_' + modeltype + '.npy', newdata)


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


def Offline(feaVec, modeltype, datatype):
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

    for i in range(tmpfea.shape[0]):
        if cnt < NUM:  # and int(tmpfea[i][-1])==0:
            newfea.append(tmpfea[i])
            cnt += 1
    for i in range(NUM2):
        newfea.append(getrandomdata(modeltype, datatype, i))

    newfea = np.array(newfea)

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
    feaVec = feaVec[newdata[:, 4].argsort()[::-1]]
    #
    #
    rank = np.array(rank)
    rank = rank[newdata[:, 4].argsort()[::-1]]
    newdata = newdata[newdata[:, 4].argsort()[::-1]]

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
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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

        feaV = np.array(feaV)
        np.save('./data/RQ1usedata/CIFAR10feature/' + datatype + modeltype + '_feature'  '.npy', feaV)

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
def Online(feaVec, modeltype, datatype):
    N = feaVec.shape[0]  # data num
    # args###################################################
    check_ratio = 0.25
    per_check = 200
    epochs = int((check_ratio * N) / per_check)
    logger.info('[datanum,check_ratio,per_check,epochs]=' + str([N, check_ratio, per_check, epochs]))
    ########################################################

    # offline part#######

    la = -1
    fea_left, sa = Offline(feaVec, modeltype, datatype)
    logger.info('success get offline result shape:' + str(fea_left.shape))

    logger.info('start online')
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




if __name__ == "__main__":
    '''
    Tips:

    You can get the features in the previous step by modifying the code(mnist_Outlier,mnist_Activation,mnist_PreLoss,mnist_mutation) 
    , which helps to reduce the algorithm complexity!

    The features are reextracted here only to demonstrate the integrity of the algorithm
    '''

    FeatureExtraction('ResNet', 'alllabel')  # get feature first

    feaVec = np.load('F:/ICSEdata/RQ1usedata/CIFAR10feature/alllabelResNet_feature.npy',
                     allow_pickle=True)

    Offline(feaVec, 'ResNet', 'alllabel')  # offline

    Online(feaVec, 'ResNet', 'alllabel')  # online

