import random
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import xlwt
from PIL import Image, ImageDraw
from torch import optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset

device = 'cuda'


def imgshow(imgpath, keypoint, bbox):
    img = Image.open('..\\MTFL\\' + imgpath)
    keypoint = keypoint.astype('float')
    bbox = bbox.astype('float')
    draw = ImageDraw.Draw(img)
    for i in range(5):
        draw.ellipse((keypoint[i] - 3, keypoint[i + 5] - 3, keypoint[i] + 3, keypoint[i + 5] + 3), 'blue')
    draw.line((bbox[0], bbox[1], bbox[0], bbox[3]), 'red')
    draw.line((bbox[0], bbox[1], bbox[2], bbox[1]), 'red')
    draw.line((bbox[2], bbox[1], bbox[2], bbox[3]), 'red')
    draw.line((bbox[0], bbox[3], bbox[2], bbox[3]), 'red')
    plt.imshow(img)
    plt.show()


def check(datatype, baseline):
    feaVec = np.load('F:/ICSEdata/RQ1data/MTFL/' + datatype + '_' + baseline + '.npy',
                     allow_pickle=True)
    oldbb = np.load('F:/ICSEdata/RQ1data/MTFL/' + datatype + '_bb_' + baseline + '.npy', allow_pickle=True)

    # feaVec = feaVec[feaVec[:, -1].argsort()[::-1]]
    # oldbb= oldbb[oldbb[:, -1].argsort()[::-1]]
    for i in range(feaVec.shape[0]):
        # if int(feaVec[i][11])==1:
        print(feaVec[i][11])
        imgshow(feaVec[i][10], feaVec[i][12:23], oldbb[i])


def getdata(orgdata, bb):
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
    imageset = torch.from_numpy(imageset)
    for index in reversed(indexes):
        # print (index)
        lm = np.delete(lm, index, axis=0)
    lm = torch.from_numpy(lm)
    return lm, imageset


def val(model, data_loader):
    model.eval()
    model.to(device)
    acc = 0
    cnt = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.to(device)
        y = y.to(device)
        out = model(X.float())
        accuracy = model.accuracy(out, y)
        acc += accuracy
        cnt += 1
    return acc / cnt


from _exp_.train_model.models import TCDCNN


def RQ3(datatype, baseline, split_ratio):
    model = TCDCNN()
    state_dict = torch.load('../models/' + datatype + '_tcdcn.pth')
    model.load_state_dict(state_dict)
    if baseline == 'offline':
        feaVec = np.load('F:/ICSEdata/RQ1data/MTFL/' + datatype + '_' + baseline + '_newclus.npy',
                         allow_pickle=True)

        oldbb = np.load('F:/ICSEdata/RQ1data/MTFL/' + datatype + '_bb_' + baseline + '_newclus.npy', allow_pickle=True)
    elif baseline == 'online':
        feaVec = np.load('F:\\ICSEdata\\online_new\\MTFL\\' + datatype + '__' + baseline + '.npy',
                         allow_pickle=True)

        oldbb = np.load('F:\\ICSEdata\\online_new\\MTFL\\' + datatype + '_bb_' + baseline + '.npy', allow_pickle=True)
    elif baseline == 'entire':
        feaVec = np.load('F:\\ICSEdata\\online_new\\MTFL\\' + datatype + '__online.npy',
                         allow_pickle=True)

        oldbb = np.load('F:\\ICSEdata\\online_new\\MTFL\\' + datatype + '_bb_online.npy', allow_pickle=True)
    data = []
    bb = []
    sum = 0
    for i in range(feaVec.shape[0]):  # 57 66
        if i <= int(feaVec.shape[0] * split_ratio):
            if int(feaVec[i][11]) == 1:
                sum += 1
                if datatype == 'D1':
                    data.append([feaVec[i][10], feaVec[i][12], feaVec[i][13], feaVec[i][14], feaVec[i][15],
                                 feaVec[i][16], feaVec[i][17], feaVec[i][18], feaVec[i][19],
                                 feaVec[i][20], feaVec[i][21]])
                    bb.append([oldbb[i][0], oldbb[i][1], oldbb[i][2], oldbb[i][3]])
                elif datatype == 'D2':
                    pass
            elif int(feaVec[i][11]) == 0:
                data.append([feaVec[i][10], feaVec[i][0], feaVec[i][1], feaVec[i][2], feaVec[i][3],
                             feaVec[i][4], feaVec[i][5], feaVec[i][6], feaVec[i][7],
                             feaVec[i][8], feaVec[i][9]])
                bb.append([oldbb[i][0], oldbb[i][1], oldbb[i][2], oldbb[i][3]])
        else:
            data.append([feaVec[i][10], feaVec[i][0], feaVec[i][1], feaVec[i][2], feaVec[i][3],
                         feaVec[i][4], feaVec[i][5], feaVec[i][6], feaVec[i][7],
                         feaVec[i][8], feaVec[i][9]])
            bb.append([oldbb[i][0], oldbb[i][1], oldbb[i][2], oldbb[i][3]])

    data = np.array(data)
    bb = np.array(bb)

    lm_orgtrain, data_orgtrain = getdata(np.loadtxt('../MTFL/training.txt', dtype=str),
                                         np.loadtxt('../MTFL/annotation.txt', dtype=str))
    orgtraindataset = TensorDataset(data_orgtrain, lm_orgtrain)
    orgtrain_loader = torch.utils.data.DataLoader(dataset=orgtraindataset, batch_size=64, shuffle=True)

    lm_orgtest, data_orgtest = getdata(np.loadtxt('../MTFL/testing1.txt', dtype=str),
                                       np.loadtxt('../MTFL/annotation_test.txt', dtype=str))
    orgtestdataset = TensorDataset(data_orgtest, lm_orgtest)
    orgtest_loader = torch.utils.data.DataLoader(dataset=orgtestdataset, batch_size=64, shuffle=True)

    trainlm, traindata = getdata(data, bb)
    traindataset = TensorDataset(traindata, trainlm)
    train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=64, shuffle=True)

    orgtrainacc = val(model, orgtrain_loader)
    orgtestacc = val(model, orgtest_loader)

    if baseline == 'offline' or baseline == 'online':
        epoch = 10
    elif baseline == 'entire':
        epoch = 30
    optim1 = optim.SGD(model.parameters(), 0.003)
    res1 = 1000
    res2 = 1000
    model.to(device)
    for e in range(0, epoch):
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            optim1.zero_grad()
            x_one = model(X.float())
            loss = model.loss([x_one],
                              [y.float()])
            loss.backward()
            optim1.step()
        nowtestacc = val(model, orgtest_loader)
        if nowtestacc < res2:
            res1 = val(model, orgtrain_loader)
            res2 = nowtestacc
    return orgtrainacc, orgtestacc, res1, res2


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


if __name__ == "__main__":

    # check(datatype='D1',baseline='online')
    dtlist = ['D1', 'D2']
    datasetname = 'MTFL'
    bllist = ['offline', 'online', 'entire']
    splist = [0.05]
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet('1')  # 在工作簿中新建一个表格
    row = -1
    for datatype in dtlist:
        row += 1
        col = -1
        for split_ratio in splist:
            for baseline in bllist:
                print('nowrun:' + datatype + ' ' + str(split_ratio) + ' ' + baseline)
                col += 2
                random.seed(6657)
                orgtrainacc, orgtestacc, res1, res2 = RQ3(datatype, baseline, split_ratio)
                if col == 1:
                    writexcel(sheet, row, col - 1, orgtrainacc, '', 0)
                    writexcel(sheet, row, col, orgtestacc, '', 0)
                writexcel(sheet, row, col + 1, res1, '', 0)
                writexcel(sheet, row, col + 2, res2, '', 0)
                workbook.save('F:/ICSEdata/excel/' + datasetname + 'fullnewclus_RQ3.xls')
