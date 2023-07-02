import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch.utils.data
from tqdm import tqdm

import matplotlib.pyplot as plt
from os.path import join
from _exp_.train_model.models import TCDCNN


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


def PreLoss(datapath, bbxpath, newdatapath, newbboxpath, model, ratio):
    model.eval()
    orgdata = np.loadtxt(datapath, dtype=str)
    bb = np.loadtxt(bbxpath, dtype=str)
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
    print("lm", lm.shape)
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
    print(indexes.shape)
    for index in reversed(indexes):
        orgdata = np.delete(orgdata, index, axis=0)
        bb = np.delete(bb, index, axis=0)

    res1 = []
    res2 = []

    with torch.no_grad():
        for i, img in tqdm(enumerate(imageset)):
            img = torch.from_numpy(img)
            out = model(img.float())
            landmark = lm[i].reshape(1, -1)

            loss = model.loss([out],
                              [landmark.float()])
            loss = loss.numpy()
            res1.append([orgdata[i][0], orgdata[i][1], orgdata[i][2], orgdata[i][3], orgdata[i][4]
                            , orgdata[i][5], orgdata[i][6], orgdata[i][7], orgdata[i][8], orgdata[i][9]
                            , orgdata[i][10], orgdata[i][11], orgdata[i][12], orgdata[i][13], orgdata[i][14]
                            , orgdata[i][15], orgdata[i][16],  loss])#orgdata[i][17],
            res2.append([bb[i][0], bb[i][1], bb[i][2], bb[i][3], bb[i][4], loss])
    res1 = np.array(res1, dtype=object)
    res2 = np.array(res2, dtype=object)
    res1 = res1[res1[:, 17].argsort()[::-1]]#18
    res2 = res2[res2[:, 5].argsort()[::-1]]
    print(res1)


    newdata = []
    sum = res1.shape[0]
    for i in range(res1.shape[0]):
        if i <= int(res1.shape[0] * ratio):
            res1[i][17] = 1
            newdata.append(res1[i])
            # if int(res1[i][15]) == 1:
            #     cnt += 1
        else:
            res1[i][17] = 0
            newdata.append(res1[i])

    print(sum)
    newdatatxt = open(newdatapath, 'w+')
    newbboxtxt = open(newbboxpath, 'w+')

    for i in range(res1.shape[0]):
        newdatatxt.write(
            str(newdata[i][0]) + ' ' + str(newdata[i][1]) + ' ' + str(newdata[i][2]) + ' ' + str(newdata[i][3])
            + ' ' + str(newdata[i][4]) + ' ' + str(newdata[i][5]) + ' ' + str(newdata[i][6]) + ' ' + str(newdata[i][7])
            + ' ' + str(newdata[i][8]) + ' ' + str(newdata[i][9]) + ' ' + str(newdata[i][10]) + ' ' + str(
                newdata[i][11])
            + ' ' + str(newdata[i][12]) + ' ' + str(newdata[i][13]) + ' ' + str(newdata[i][14]) + ' ' + str(
                newdata[i][15])
            + ' ' + str(newdata[i][16]) + ' ' + str(newdata[i][17])  + '\n')#+ ' ' + str(newdata[i][18])

        newbboxtxt.write(
            str(res2[i][0]) + ' ' + str(res2[i][1]) + ' ' + str(res2[i][2]) + ' ' + str(res2[i][3]) + ' ' + str(
                res2[i][4]) + '\n')
    newdatatxt.close()
    newbboxtxt.close()


if __name__ == "__main__":
    datapath = '../MTFL/TCDCN_NEWDATA_MID/training_org_VAE_MID_new.txt'
    bbxpath = '../MTFL/TCDCN_NEWDATA_MID/annotation_org_VAE_MID_new.txt'
    newdatapath = '../MTFL/TCDCN_NEWDATA_OUT/training_org_VAE_MID_OUT_new.txt'
    newbboxpath = '../MTFL/TCDCN_NEWDATA_OUT/annotation_org_VAE_MID_OUT_new.txt'

    modelpath = '../models/org_tcdcn.pth'
    model = TCDCNN()

    state_dict = torch.load(modelpath)
    model.load_state_dict(state_dict)

    PreLoss(datapath, bbxpath, newdatapath, newbboxpath, model, 0.05)
