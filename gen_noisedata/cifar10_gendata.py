import argparse
import os
import random
import cv2
import numpy as np
from cv2 import imwrite
from keras.datasets import cifar10

parser = argparse.ArgumentParser()
'''
Read the following tips before running:

1.Save the cifar-10 data to a folder according to the label category first.

2.When generate different type datas, using datapath=the folder generated above.
'''
parser.add_argument('--datapath', type=str, default='./data/CIFA10/CIFA10_PNG/', help='input data path.')
parser.add_argument('--savepath', type=str, default='./data/', help='save data path.')
parser.add_argument('--datatype', type=str, default='original', help='datatype to generate which includes original or '
                                                                     'RandomLabelNoise or SpecificLabelNoise'
                                                                     'RandomDataNoise or SpecificDataNoise '
                                                                     'you shall generate original data first.')
parser.add_argument('--noisedatapath', type=str, default='./data/MNIST/MNIST_PNG/train/', help='Noise data path.')
parser.add_argument('--ratio', type=float, default=0.05, help='The proportion of noisy data.')
args = parser.parse_args()

def load_orgdata(datapth, savepath):
    traindata = []
    for label in os.listdir(datapth + 'train'):
        for imgname in os.listdir(datapth + 'train/' + str(label)):
            imgpath = datapth + 'train/' + str(label) + '/' + imgname

            img = cv2.imread(imgpath)


            traindata.append([label, img, 0])
    traindata = np.array(traindata, dtype=object)
    print("train data shape: ", traindata.shape)
    np.save(savepath + 'orgtraindata', traindata)

    testdata = []
    for label in os.listdir(datapth + 'test'):
        for imgname in os.listdir(datapth + 'test/' + str(label)):
            imgpath = datapth + 'test/' + str(label) + '/' + imgname

            img = cv2.imread(imgpath)


            testdata.append([label, img, 0])
    testdata = np.array(testdata, dtype=object)
    print("test data shape: ", testdata.shape)
    np.save(savepath + 'orgtestdata', testdata)

def load_alllabeldata(datapth, savepath, ratio):
    traindata = []
    trainNumAll = [5000,5000,5000,5000,5000,5000,5000,5000,5000,5000]
    testNumAll = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
    for label in os.listdir(datapth + 'train'):
        cnt = 0
        for imgname in os.listdir(datapth + 'train/' + str(label)):
            if cnt < trainNumAll[int(label)] * ratio:
                imgpath = datapth + 'train/' + str(label) + '/' + imgname
                img = cv2.imread(imgpath)

                newlabel = random.randint(0, 9)
                while newlabel == int(label):
                    newlabel = random.randint(0, 9)
                traindata.append([newlabel, img, 1])
                cnt += 1
            else:
                imgpath = datapth + 'train/' + str(label) + '/' + imgname
                img = cv2.imread(imgpath)

                traindata.append([label, img, 0])
        print(cnt)
    traindata = np.array(traindata, dtype=object)
    print("train data shape: ", traindata.shape)
    np.save(savepath + 'alllabeltraindata', traindata)

    testdata = []
    for label in os.listdir(datapth + 'test'):
        cnt = 0
        for imgname in os.listdir(datapth + 'test/' + str(label)):
            if cnt < testNumAll[int(label)] * ratio:
                imgpath = datapth + 'test/' + str(label) + '/' + imgname
                img = cv2.imread(imgpath)

                newlabel = random.randint(0, 9)
                while newlabel == int(label):
                    newlabel = random.randint(0, 9)
                testdata.append([newlabel, img, 1])
                cnt += 1
            else:
                imgpath = datapth + 'test/' + str(label) + '/' + imgname
                img = cv2.imread(imgpath)

                testdata.append([label, img, 0])
        print(cnt)
    testdata = np.array(testdata, dtype=object)
    print("test data shape: ", testdata.shape)
    np.save(savepath + 'alllabeltestdata', testdata)


def load_ranlabeldata(datapth, savepath, ratio):
    traindata = []
    trainNumAll = [5000,5000,5000,5000,5000,5000,5000,5000,5000,5000]
    testNumAll = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
    swl = random.sample(range(10), 2)
    print("swap list: ", swl)
    cnt0 = 0
    cnt1 = 0
    for label in os.listdir(datapth + 'train'):
        for imgname in os.listdir(datapth + 'train/' + str(label)):
            if int(label) == swl[0] and cnt0 < trainNumAll[int(label)] * ratio:
                imgpath = datapth + 'train/' + str(label) + '/' + imgname
                img = cv2.imread(imgpath)

                newlabel = swl[1]
                traindata.append([newlabel, img, 1])
                cnt0 += 1
            elif int(label) == swl[1] and cnt1 < trainNumAll[int(label)] * ratio:
                imgpath = datapth + 'train/' + str(label) + '/' + imgname
                img = cv2.imread(imgpath)

                newlabel = swl[0]
                traindata.append([newlabel, img, 1])
                cnt1 += 1
            else:
                imgpath = datapth + 'train/' + str(label) + '/' + imgname
                img = cv2.imread(imgpath)

                traindata.append([label, img, 0])
    print(cnt0,cnt1)
    traindata = np.array(traindata, dtype=object)
    print("train data shape: ", traindata.shape)
    np.save(savepath + 'ranlabeltraindata', traindata)

    testdata = []
    cnt0=0
    cnt1=0
    for label in os.listdir(datapth + 'test'):
        for imgname in os.listdir(datapth + 'test/' + str(label)):
            if int(label) == swl[0] and cnt0 < testNumAll[int(label)] * ratio:
                imgpath = datapth + 'test/' + str(label) + '/' + imgname
                img = cv2.imread(imgpath)

                newlabel = swl[1]
                testdata.append([newlabel, img, 1])
                cnt0 += 1
            elif int(label) == swl[1] and cnt1 < testNumAll[int(label)] * ratio:
                imgpath = datapth + 'test/' + str(label) + '/' + imgname
                img = cv2.imread(imgpath)

                newlabel = swl[0]
                testdata.append([newlabel, img, 1])
                cnt1 += 1

            else:
                imgpath = datapth + 'test/' + str(label) + '/' + imgname
                img = cv2.imread(imgpath)

                testdata.append([label, img, 0])
    print(cnt0,cnt1)
    testdata = np.array(testdata, dtype=object)
    print("test data shape: ", testdata.shape)
    np.save(savepath + 'ranlabeltestdata', testdata)


def load_alldirtydata(datapth, savepath, ratio):
    traindata = []
    trainNumAll = [5000,5000,5000,5000,5000,5000,5000,5000,5000,5000]
    testNumAll = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
    for label in os.listdir(datapth + 'train'):
        cnt = 0
        for imgname in os.listdir(datapth + 'train/' + str(label)):
            if cnt < trainNumAll[int(label)] * ratio:
                newimg_path = args.noisedatapath
                p = random.randint(0, 9)
                newimg_path = newimg_path + str(p)
                list = []
                for x in os.listdir(newimg_path):
                    list.append(x)
                imagename = random.sample(list, 1)
                newimg = cv2.imread(newimg_path + "/" + str(imagename[0]))
                newimg = cv2.resize(newimg, (32, 32), interpolation=cv2.INTER_CUBIC)

                traindata.append([label, newimg, 1])
                cnt += 1
            else:
                imgpath = datapth + 'train/' + str(label) + '/' + imgname
                img = cv2.imread(imgpath)

                traindata.append([label, img, 0])
        print(cnt)
    traindata = np.array(traindata, dtype=object)
    print("train data shape: ", traindata.shape)
    np.save(savepath + 'alldirtytraindata', traindata)

    testdata = []
    for label in os.listdir(datapth + 'test'):
        cnt = 0
        for imgname in os.listdir(datapth + 'test/' + str(label)):
            if cnt < testNumAll[int(label)] * ratio:
                newimg_path = args.noisedatapath
                p = random.randint(0, 9)
                newimg_path = newimg_path + str(p)
                list = []
                for x in os.listdir(newimg_path):
                    list.append(x)
                imagename = random.sample(list, 1)
                newimg = cv2.imread(newimg_path + "/" + str(imagename[0]))
                newimg = cv2.resize(newimg, (32, 32), interpolation=cv2.INTER_CUBIC)

                testdata.append([label, newimg, 1])
                cnt += 1
            else:
                imgpath = datapth + 'test/' + str(label) + '/' + imgname
                img = cv2.imread(imgpath)

                testdata.append([label, img, 0])
    testdata = np.array(testdata, dtype=object)
    print("test data shape: ", testdata.shape)
    np.save(savepath + 'alldirtytestdata', testdata)


def load_randirtydata(datapth, savepath, ratio):
    traindata = []
    trainNumAll = [5000,5000,5000,5000,5000,5000,5000,5000,5000,5000]
    testNumAll = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
    dirtylabel = random.randint(0, 9)
    print("dirty label: ", dirtylabel)
    cnt = 0
    for label in os.listdir(datapth + 'train'):
        for imgname in os.listdir(datapth + 'train/' + str(label)):
            if int(dirtylabel) == int(label) and cnt < trainNumAll[int(label)] * ratio:
                newimg_path = args.noisedatapath
                p = random.randint(0, 9)
                newimg_path = newimg_path + str(p)
                list = []
                for x in os.listdir(newimg_path):
                    list.append(x)
                imagename = random.sample(list, 1)
                newimg = cv2.imread(newimg_path + "/" + str(imagename[0]))
                newimg = cv2.resize(newimg, (32, 32), interpolation=cv2.INTER_CUBIC)

                traindata.append([label, newimg, 1])
                cnt += 1
            else:
                imgpath = datapth + 'train/' + str(label) + '/' + imgname
                img = cv2.imread(imgpath)

                traindata.append([label, img, 0])
    print(cnt)
    traindata = np.array(traindata, dtype=object)
    print("train data shape: ", traindata.shape)
    np.save(savepath + 'randirtytraindata', traindata)

    testdata = []
    cnt = 0
    for label in os.listdir(datapth + 'test'):
        for imgname in os.listdir(datapth + 'test/' + str(label)):
            if int(dirtylabel) == int(label) and cnt < testNumAll[int(label)] * ratio:
                newimg_path = args.noisedatapath
                p = random.randint(0, 9)
                newimg_path = newimg_path + str(p)
                list = []
                for x in os.listdir(newimg_path):
                    list.append(x)
                imagename = random.sample(list, 1)
                newimg = cv2.imread(newimg_path + "/" + str(imagename[0]))
                newimg = cv2.resize(newimg, (32, 32), interpolation=cv2.INTER_CUBIC)

                testdata.append([label, newimg, 1])
                cnt += 1
            else:
                imgpath = datapth + 'test/' + str(label) + '/' + imgname
                img = cv2.imread(imgpath)

                testdata.append([label, img, 0])
    print(cnt)
    testdata = np.array(testdata, dtype=object)
    print("test data shape: ", testdata.shape)
    np.save(savepath + 'randirtytestdata', testdata)

if __name__ == "__main__":
    if args.datatype == 'original':
        load_orgdata(datapth=args.datapath, savepath=args.savepath)
    if args.datatype == 'RandomLabelNoise':
        load_alllabeldata(datapth=args.datapath, savepath=args.savepath, ratio=args.ratio)
    if args.datatype == 'SpecificLabelNoise':
        load_ranlabeldata(datapth=args.datapath, savepath=args.savepath, ratio=args.ratio)
    if args.datatype == 'RandomDataNoise':
        load_alldirtydata(datapth=args.datapath, savepath=args.savepath, ratio=args.ratio)
    if args.datatype == 'SpecificDataNoise':
        load_randirtydata(datapth=args.datapath, savepath=args.savepath, ratio=args.ratio)
