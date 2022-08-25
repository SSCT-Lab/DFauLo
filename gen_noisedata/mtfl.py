import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from os.path import isfile, join

from PIL import Image, ImageDraw
parser = argparse.ArgumentParser()

'''
Read the following tips before running:

1.Download MTFL dataset in a folder and use any face detction tool to get bounding boxes.

2.When generate different type datas, using datapath=the folder generated above.
'''
parser.add_argument('--datapath', type=str, default='./data/MTFL/training.txt', help='input data path.')
parser.add_argument('--bbxpath', type=str, default='./data/MTFL/annotation.txt', help='input bounding box path.')
parser.add_argument('--savedatapath', type=str, default='./data/MTFL/training_D1.txt', help='save data path, noise types should be distinguished.')
parser.add_argument('--savebbxpath', type=str, default='./data/MTFL/annotation_D1.txt', help='save bounding boxes path.')
parser.add_argument('--noisedatapath', type=str, default='./data/COCO/val2017', help='Noise data path.')
parser.add_argument('--ratio', type=float, default=0.05, help='The proportion of noisy data.')


args = parser.parse_args()
def imgshow(imgpath, keypoint, bbox):
    img = Image.open(imgpath)
    keypoint = keypoint.astype('float')
    draw = ImageDraw.Draw(img)
    for i in range(5):
        draw.ellipse((keypoint[i] - 3, keypoint[i + 5] - 3, keypoint[i] + 3, keypoint[i + 5] + 3), 'blue')
    draw.line((bbox[0], bbox[1], bbox[0], bbox[3]), 'red')
    draw.line((bbox[0], bbox[1], bbox[2], bbox[1]), 'red')
    draw.line((bbox[2], bbox[1], bbox[2], bbox[3]), 'red')
    draw.line((bbox[0], bbox[3], bbox[2], bbox[3]), 'red')
    plt.imshow(img)
    plt.show()


def genD1data(orgdatapath, orgbbxpath, newdatapath, newbboxpath,ratio):
    orgtrain = np.loadtxt(orgdatapath, dtype=str)
    orgtrainbbx = np.loadtxt(orgbbxpath)
    print(orgtrain.shape)
    list = []
    for i in range(orgtrain.shape[0]):
        list.append(i)
    ranlist = random.sample(list, int(orgtrain.shape[0] * ratio))
    ranlist = np.array(ranlist)
    print(ranlist.shape)
    newdatatxt = open(newdatapath, 'w+')
    newbboxtxt = open(newbboxpath, 'w+')

    for i in ranlist:
        imgpath = join('./data/MTFL/', orgtrain[i][0])
        keypoint = orgtrain[i][1:11]
        bbox = orgtrainbbx[i]

        # imgshow(imgpath=imgpath,keypoint=keypoint,bbox=bbox)

        xmax = max(bbox[0], bbox[2])
        xmin = min(bbox[0], bbox[2])
        ymax = max(bbox[1], bbox[3])
        ymin = min(bbox[1], bbox[3])

        newkeyx = np.random.randint(xmin, xmax, 5)
        newkeyy = np.random.randint(ymin, ymax, 5)

        newkeypoint = np.concatenate((newkeyx, newkeyy), axis=0)
        # imgshow(imgpath=imgpath, keypoint=newkeypoint, bbox=bbox)

        newdatatxt.write(
            orgtrain[i][0] + ' ' + str(newkeypoint[0]) + ' ' + str(newkeypoint[1]) + ' ' + str(newkeypoint[2]) + ' '
            + str(newkeypoint[3]) + ' ' + str(newkeypoint[4]) + ' ' + str(newkeypoint[5]) + ' ' + str(
                newkeypoint[6]) + ' '
            + str(newkeypoint[7]) + ' ' + str(newkeypoint[8]) + ' ' + str(newkeypoint[9]) + ' '
            + str(orgtrain[i][11]) + ' ' + str(orgtrain[i][12]) + ' ' + str(orgtrain[i][13]) + ' ' + str(
                orgtrain[i][14]) + ' ' + '1\n')
        newbboxtxt.write(
            str(orgtrainbbx[i][0]) + ' ' + str(orgtrainbbx[i][1]) + ' ' + str(orgtrainbbx[i][2]) + ' ' + str(
                orgtrainbbx[i][3]) + ' ' + str(orgtrainbbx[i][4]) + '\n')

    for i in range(orgtrain.shape[0]):
        if i in ranlist:
            continue
        newdatatxt.write(
            orgtrain[i][0] + ' ' + str(orgtrain[i][1]) + ' ' + str(orgtrain[i][2]) + ' ' + str(orgtrain[i][3]) + ' '
            + str(orgtrain[i][4]) + ' ' + str(orgtrain[i][5]) + ' ' + str(orgtrain[i][6]) + ' ' + str(
                orgtrain[i][7]) + ' ' + str(orgtrain[i][8]) + ' ' + str(orgtrain[i][9]) + ' ' + str(
                orgtrain[i][10]) + ' '
            + str(orgtrain[i][11]) + ' ' + str(orgtrain[i][12]) + ' ' + str(orgtrain[i][13]) + ' ' + str(
                orgtrain[i][14]) + ' ' + '0\n')
        newbboxtxt.write(
            str(orgtrainbbx[i][0]) + ' ' + str(orgtrainbbx[i][1]) + ' ' + str(orgtrainbbx[i][2]) + ' ' + str(
                orgtrainbbx[i][3]) + ' ' + str(orgtrainbbx[i][4]) + '\n')

    newdatatxt.close()


def genD2data(orgdatapath, orgbbxpath, newdatapath, newbboxpath,ratio):
    orgtrain = np.loadtxt(orgdatapath, dtype=str)
    orgtrainbbx = np.loadtxt(orgbbxpath)
    print(orgtrain.shape)
    list = []
    for i in range(orgtrain.shape[0]):
        list.append(i)
    ranlist = random.sample(list, int(orgtrain.shape[0] * ratio))
    ranlist = np.array(ranlist)
    print(ranlist.shape)
    newdatatxt = open(newdatapath, 'w+')
    newbboxtxt = open(newbboxpath, 'w+')
    newimglst = []
    for x in os.listdir(args.noisedatapath):
        newimglst.append(x)

    for i in ranlist:
        imgpath = join('./data/MTFL/', orgtrain[i][0])
        keypoint = orgtrain[i][1:11]
        bbox = orgtrainbbx[i]
        newimgpath = random.sample(newimglst, 1)
        newimgpath = 'val2017\\' + newimgpath[0]
        img = Image.open(args.noisedatapath + newimgpath)

        xmax = img.width
        ymax = img.height

        newkeyx = np.random.randint(0, xmax, 5)
        newkeyy = np.random.randint(0, ymax, 5)
        newkeypoint = np.concatenate((newkeyx, newkeyy), axis=0)
        newbbox = [0, 0, xmax, ymax, 0.99]

        # imgshow(imgpath,keypoint,bbox)
        # imgshow('../MTFL/'+newimgpath,newkeypoint,newbbox)

        newdatatxt.write(
            newimgpath + ' ' + str(newkeypoint[0]) + ' ' + str(newkeypoint[1]) + ' ' + str(newkeypoint[2]) + ' '
            + str(newkeypoint[3]) + ' ' + str(newkeypoint[4]) + ' ' + str(newkeypoint[5]) + ' ' + str(
                newkeypoint[6]) + ' '
            + str(newkeypoint[7]) + ' ' + str(newkeypoint[8]) + ' ' + str(newkeypoint[9]) + ' '
            + str(orgtrain[i][11]) + ' ' + str(orgtrain[i][12]) + ' ' + str(orgtrain[i][13]) + ' ' + str(
                orgtrain[i][14]) + ' ' + '1\n')

        newbboxtxt.write(
            str(newbbox[0]) + ' ' + str(newbbox[1]) + ' ' + str(newbbox[2]) + ' ' + str(newbbox[3]) + ' ' + str(
                newbbox[4]) + '\n')

    for i in range(orgtrain.shape[0]):
        if i in ranlist:
            continue
        newdatatxt.write(
            orgtrain[i][0] + ' ' + str(orgtrain[i][1]) + ' ' + str(orgtrain[i][2]) + ' ' + str(orgtrain[i][3]) + ' '
            + str(orgtrain[i][4]) + ' ' + str(orgtrain[i][5]) + ' ' + str(orgtrain[i][6]) + ' ' + str(
                orgtrain[i][7]) + ' ' + str(orgtrain[i][8]) + ' ' + str(orgtrain[i][9]) + ' ' + str(
                orgtrain[i][10]) + ' '
            + str(orgtrain[i][11]) + ' ' + str(orgtrain[i][12]) + ' ' + str(orgtrain[i][13]) + ' ' + str(
                orgtrain[i][14]) + ' ' + '0\n')
        newbboxtxt.write(
            str(orgtrainbbx[i][0]) + ' ' + str(orgtrainbbx[i][1]) + ' ' + str(orgtrainbbx[i][2]) + ' ' + str(
                orgtrainbbx[i][3]) + ' ' + str(orgtrainbbx[i][4]) + '\n')

    newdatatxt.close()
    newbboxtxt.close()
if __name__ == "__main__":

    if args.datatype == 'LabelNoise':
        genD1data(orgdatapath=args.datapath, orgbbxpath=args.bbxpath, newdatapath=args.savedatapath, newbboxpath=args.savebbxpath,ratio=args.ratio)

    if args.datatype == 'DataNoise':
        genD2data(orgdatapath=args.datapath, orgbbxpath=args.bbxpath, newdatapath=args.savedatapath, newbboxpath=args.savebbxpath,ratio=args.ratio)
