import cv2
import numpy as np
from pyod.models.vae import VAE
from tqdm import tqdm

from os.path import join


def Outlier(datapath, bbxpath, newdatapath, newbboxpath, ratio):
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
        imageset.append(resized)
    x_train = np.array(imageset)
    x_train = x_train.reshape(x_train.shape[0], 40 * 40)
    print(x_train.shape)
    indexes = np.array(indexes)

    for index in reversed(indexes):
        orgdata = np.delete(orgdata, index, axis=0)
        bb = np.delete(bb, index, axis=0)

    clf = VAE(epochs=5)
    clf.fit(x_train)
    y_score = clf.decision_scores_
    y_label = clf.labels_
    y_label = np.array(y_label)
    y_score = np.array(y_score)

    res1 = []
    res2 = []
    for i in range(orgdata.shape[0]):
        res1.append([orgdata[i][0], orgdata[i][1], orgdata[i][2], orgdata[i][3], orgdata[i][4]
                        , orgdata[i][5], orgdata[i][6], orgdata[i][7], orgdata[i][8], orgdata[i][9]
                        , orgdata[i][10], orgdata[i][11], orgdata[i][12], orgdata[i][13], orgdata[i][14]
                        , y_score[i]])
        res2.append([bb[i][0], bb[i][1], bb[i][2], bb[i][3], bb[i][4], y_score[i]])
    res1 = np.array(res1, dtype=object)
    res2 = np.array(res2, dtype=object)

    cnt = 0
    res1 = res1[res1[:, 15].argsort()[::-1]]
    res2 = res2[res2[:, 5].argsort()[::-1]]


    newdata = []
    sum = res1.shape[0]
    for i in range(res1.shape[0]):
        if i <= int(res1.shape[0] * ratio):
            res1[i][15] = 1
            newdata.append(res1[i])
            if int(res1[i][15]) == 1:
                cnt += 1
        else:
            res1[i][15] = 0
            newdata.append(res1[i])

    print(cnt / (sum * ratio))
    newdatatxt = open(newdatapath, 'w+')
    newbboxtxt = open(newbboxpath, 'w+')

    for i in range(res1.shape[0]):
        newdatatxt.write(
            str(newdata[i][0]) + ' ' + str(newdata[i][1]) + ' ' + str(newdata[i][2]) + ' ' + str(newdata[i][3])
            + ' ' + str(newdata[i][4]) + ' ' + str(newdata[i][5]) + ' ' + str(newdata[i][6]) + ' ' + str(newdata[i][7])
            + ' ' + str(newdata[i][8]) + ' ' + str(newdata[i][9]) + ' ' + str(newdata[i][10]) + ' ' + str(
                newdata[i][11])
            + ' ' + str(newdata[i][12]) + ' ' + str(newdata[i][13]) + ' ' + str(newdata[i][14]) + ' ' + str(
                newdata[i][15])+'\n')

        newbboxtxt.write(
            str(res2[i][0]) + ' ' + str(res2[i][1]) + ' ' + str(res2[i][2]) + ' ' + str(res2[i][3]) + ' ' + str(
                res2[i][4]) + '\n')
    newdatatxt.close()
    newbboxtxt.close()


if __name__ == "__main__":
    datapath = '../MTFL/training.txt'
    bbxpath = '../MTFL/annotation.txt'
    newdatapath = '../MTFL/TCDCN_NEWDATA_IN/training_org_VAE.txt'
    newbboxpath = '../MTFL/TCDCN_NEWDATA_IN/annotation_org_VAE.txt'
    # newdatapath=''

    Outlier(datapath, bbxpath, newdatapath, newbboxpath, 0.05)

