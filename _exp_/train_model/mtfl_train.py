import argparse

import torch.autograd as A
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.utils.data
import torch.optim as op
import math

from tqdm import tqdm

from tcdcnn_dataset import FaceLandmarksDataset
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
import sys
import cv2
from models import TCDCNN

net = TCDCNN()
parser = argparse.ArgumentParser()

parser.add_argument('--mypath', type=str, default="./data/MTFL", help='dataset rootpath.')
parser.add_argument('--traindata', type=str, default="training.txt", help='traindata.')
parser.add_argument('--annotation_train', type=str, default="annotation.txt", help='annotation_train.')
parser.add_argument('--testdata', type=str, default="testing1.txt", help='testdata.')
parser.add_argument('--annotation_test', type=str, default="annotation_test.txt", help='annotation_test.')
parser.add_argument('--model_savepath', type=str, default="./models/MTFL.pth", help='model_savepath.')
args = parser.parse_args()



def test_model(model, dataloader_test):
    for i, data in enumerate(dataloader_test, 1):
        images, landmark, gender, smile, glass, pose = data
        landmark = A.Variable(landmark)
        images = A.Variable(images)

        x_one = net(images.float())

        accuracy = net.accuracy(x_one, landmark.float())
        return accuracy

def train(mypath,traindata,annotation_train,testdata,annotation_test,model_savepath):

    optim = op.SGD(net.parameters(), 0.003)

    faceLandMark = FaceLandmarksDataset(mypath, traindata, annotation_train)
    dataloader = DataLoader(faceLandMark, batch_size=64,
                            shuffle=True, num_workers=0)
    faceLandMark_test = FaceLandmarksDataset(mypath, testdata, annotation_test)
    dataloader_test = DataLoader(faceLandMark_test, batch_size=64,
                                 shuffle=True, num_workers=0)

    print("Starting training")


    epochs = 40

    epo_list = []

    for e in range(0, epochs):
        epo_list.append(e)
        total_accuracy = 0
        loss_total = 0
        count = 0
        total_accuracy_training = 0
        for i, data in tqdm(enumerate(dataloader, 1)):
            images, landmark, gender, smile, glass, pose = data

            landmark = A.Variable(landmark)

            images_temp = images.clone()
            optim.zero_grad()
            x_one = net(images.float())
            loss = net.loss([x_one],
                            [landmark.float()])
            loss.backward()
            optim.step()
            # loss for training
            loss_total = loss_total + loss.item()
            accuracy_training = net.accuracy(x_one, landmark.float())
            count = count + 1
            loss_text = "Loss:{}".format(float(loss_total / count))
            # logging.info(loss_text)
            # Accuracy for testing
            net.eval()
            accuracy = test_model(net, dataloader_test)
            total_accuracy = total_accuracy + accuracy
            # Accuracy for training
            total_accuracy_training = total_accuracy_training + accuracy_training
            net.train()

        print("loss: ", loss_total / count)
        print("test MNE rate: ", total_accuracy / count)
        print("train MNE ratio:", total_accuracy_training / count)

    torch.save(net.state_dict(), model_savepath)
if __name__ == "__main__":
    train(args.mypath, args.traindata, args.annotation_train, args.testdata,
          args.annotation_test, args.model_savepath)
