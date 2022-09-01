import time

import cv2
import torch.autograd as A
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import math
from pyod.models.vae import VAE
from tqdm import tqdm
from sklearn.cluster import KMeans
from train_model.tcdcnn_dataset import FaceLandmarksDataset
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
import sys
from os.path import isfile, join
from train_model.models import TCDCNN

def imgshow(img, keypoint, bbox):
    img = img.reshape(40, 40)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for i in range(5):
        draw.ellipse((keypoint[i] - 3, keypoint[i + 5] - 3, keypoint[i] + 3, keypoint[i + 5] + 3), 'blue')
    draw.line((float(bbox[0]), float(bbox[1]), float(bbox[0]), float(bbox[3])), 'red')
    draw.line((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[1])), 'red')
    draw.line((float(bbox[2]), float(bbox[1]), float(bbox[2]), float(bbox[3])), 'red')
    draw.line((float(bbox[0]), float(bbox[3]), float(bbox[2]), float(bbox[3])), 'red')
    plt.imshow(img)
    plt.show()


def cluster(datapath, bbxpath, newdatapath, newbboxpath, model,ratio):
    orgdata = np.loadtxt(datapath, dtype=str)
    bb = np.loadtxt(bbxpath, dtype=str)

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

    x_train = []
    with torch.no_grad():
        for img in tqdm(imageset):
            img = torch.from_numpy(img)
            out = model.features(img.float())

            x_train.append(out.numpy()[0])

    x_train = np.array(x_train)
    print(x_train.shape)

    clf = KMeans(n_clusters=2)
    clf.fit(x_train)
    y_label = clf.labels_
    centers = clf.cluster_centers_
    y_label = np.array(y_label)

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
        seldata = 0
        lessdata = 1
    else:
        seldata = 1
        lessdata = 0

    res1 = []
    res2 = []
    for i in range(orgdata.shape[0]):
        if y_label[i] == seldata:
            res1.append([orgdata[i][0], orgdata[i][1], orgdata[i][2], orgdata[i][3], orgdata[i][4]
                            , orgdata[i][5], orgdata[i][6], orgdata[i][7], orgdata[i][8], orgdata[i][9]
                            , orgdata[i][10], orgdata[i][11], orgdata[i][12], orgdata[i][13], orgdata[i][14]
                            , orgdata[i][15],0])# orgdata[i][16]
            res2.append([bb[i][0], bb[i][1], bb[i][2], bb[i][3], bb[i][4], 0])
        else:
            res1.append([orgdata[i][0], orgdata[i][1], orgdata[i][2], orgdata[i][3], orgdata[i][4]
                            , orgdata[i][5], orgdata[i][6], orgdata[i][7], orgdata[i][8], orgdata[i][9]
                            , orgdata[i][10], orgdata[i][11], orgdata[i][12], orgdata[i][13], orgdata[i][14]
                            , orgdata[i][15], 1])#orgdata[i][16],
            res2.append([bb[i][0], bb[i][1], bb[i][2], bb[i][3], bb[i][4], 1])
    res1 = np.array(res1, dtype=object)
    res2 = np.array(res2, dtype=object)


    newdatatxt = open(newdatapath, 'w+')
    newbboxtxt = open(newbboxpath, 'w+')

    for i in range(res1.shape[0]):
        newdatatxt.write(
            str(res1[i][0]) + ' ' + str(res1[i][1]) + ' ' + str(res1[i][2]) + ' ' + str(res1[i][3])
            + ' ' + str(res1[i][4]) + ' ' + str(res1[i][5]) + ' ' + str(res1[i][6]) + ' ' + str(res1[i][7])
            + ' ' + str(res1[i][8]) + ' ' + str(res1[i][9]) + ' ' + str(res1[i][10]) + ' ' + str(
                res1[i][11])
            + ' ' + str(res1[i][12]) + ' ' + str(res1[i][13]) + ' ' + str(res1[i][14]) + ' ' + str(
                res1[i][15])
            + ' ' + str(res1[i][16]) +'\n')#+' '+ str(res1[i][17])

        newbboxtxt.write(
            str(res2[i][0]) + ' ' + str(res2[i][1]) + ' ' + str(res2[i][2]) + ' ' + str(res2[i][3]) + ' ' + str(
                res2[i][4]) + '\n')
    newdatatxt.close()
    newbboxtxt.close()


if __name__ == "__main__":
    datapath = '../MTFL/TCDCN_NEWDATA_IN/training_org_VAE.txt'
    bbxpath = '../MTFL/TCDCN_NEWDATA_IN/annotation_org_VAE.txt'
    newdatapath = '../MTFL/TCDCN_NEWDATA_MID/training_org_VAE_MID_new.txt'
    newbboxpath = '../MTFL/TCDCN_NEWDATA_MID/annotation_org_VAE_MID_new.txt'
    modelpath = '../models/org_tcdcn.pth'
    model = TCDCNN()

    state_dict = torch.load(modelpath)
    model.load_state_dict(state_dict)

    cluster(datapath, bbxpath, newdatapath, newbboxpath, model,0.05)
