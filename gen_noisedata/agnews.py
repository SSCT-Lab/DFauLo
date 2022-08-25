import argparse
import pickle
import random
import time

import numpy as np
import pandas as pd
from myvocab import Vocab

from torchtext.data.utils import get_tokenizer

parser = argparse.ArgumentParser()
'''
Read the following tips before running:

1.Download AgNews dataset: train.csv & test.csv in a folder.

2.When generate different type datas, using datapath=the folder generated above.
'''
parser.add_argument('--trainpath', type=str, default='./data/AgNews/AgNews_Raw/train.csv', help='input train data path.')
parser.add_argument('--testpath', type=str, default='./data/AgNews/AgNews_Raw/test.csv', help='input test data path.')
parser.add_argument('--savepath', type=str, default='./data/', help='save data path.')
parser.add_argument('--vocabpath', type=str, default='vocab.pkl', help='vocab path.')


parser.add_argument('--datatype', type=str, default='original', help='datatype to generate which includes original or '
                                                                     'RandomLabelNoise or SpecificLabelNoise'
                                                                     'RandomDataNoise or SpecificDataNoise ')

parser.add_argument('--noisedatapath', type=str, default='./data/AgNews/AgNews_Raw/imdb_master.csv', help='Noise data path.')
parser.add_argument('--ratio', type=float, default=0.05, help='The proportion of noisy data.')
args = parser.parse_args()


vocab = pickle.load(open(args.vocabpath, "rb"))

device = "cuda"

def save_org_data(trainpath, testpath, savepath):
    df = pd.read_csv(trainpath)
    traindata = []
    for i in range(len(df)):
        traindata.append([int(df['Class Index'][i]) - 1, df['Title'][i] + ' ' + df['Description'][i], int(0)])
    traindata = np.array(traindata)
    np.save(savepath + 'orgtraindata', traindata)
    print("train data shape: ", traindata.shape)

    df2 = pd.read_csv(testpath)
    testdata = []
    for i in range(len(df2)):
        testdata.append([int(df2['Class Index'][i]) - 1, df2['Title'][i] + ' ' + df2['Description'][i], int(0)])
    testdata = np.array(testdata)
    np.save(savepath + 'orgtestdata', testdata)
    print("test data shape: ", testdata.shape)


def save_alllabel_data(trainpath, testpath, savepath, ratio):
    df = pd.read_csv(trainpath)
    trainNumAll = [30000, 30000, 30000, 30000]
    testNumAll = [1900, 1900, 1900, 1900]
    cnttrain = [0, 0, 0, 0]
    traindata = []
    for i in range(len(df)):
        traindata.append([int(df['Class Index'][i]) - 1, df['Title'][i] + ' ' + df['Description'][i], int(0)])
    traindata = np.array(traindata)

    alllabeldatatrain = []
    for i in range(120000):
        lb = int(traindata[i][0])
        contxt = traindata[i][1]
        flag = traindata[i][2]
        if cnttrain[lb] < trainNumAll[lb] * ratio:
            newlabel = random.randint(0, 3)
            while newlabel == int(lb):
                newlabel = random.randint(0, 3)
            alllabeldatatrain.append([newlabel, contxt, 1])
            cnttrain[lb] += 1
        else:
            alllabeldatatrain.append([lb, contxt, 0])
    alllabeldatatrain = np.array(alllabeldatatrain)
    np.save(savepath + 'alllabeltraindata', alllabeldatatrain)
    print("train data shape: ", alllabeldatatrain.shape)

    cnttest = [0, 0, 0, 0]
    df2 = pd.read_csv(testpath)
    testdata = []
    for i in range(len(df2)):
        testdata.append([int(df2['Class Index'][i]) - 1, df2['Title'][i] + ' ' + df2['Description'][i], int(0)])
    testdata = np.array(testdata)

    alllabeldatatest = []
    for i in range(7600):
        lb = int(testdata[i][0])
        contxt = testdata[i][1]
        flag = testdata[i][2]
        if cnttest[lb] < testNumAll[lb] * ratio:
            newlabel = random.randint(0, 3)
            while newlabel == int(lb):
                newlabel = random.randint(0, 3)
            alllabeldatatest.append([newlabel, contxt, 1])
            cnttest[lb] += 1
        else:
            alllabeldatatest.append([lb, contxt, 0])
    alllabeldatatest = np.array(alllabeldatatest)
    np.save(savepath + 'alllabeltestdata', alllabeldatatest)
    print("test data shape: ", alllabeldatatest.shape)


def save_ranlabel_data(trainpath, testpath, savepath, ratio):
    df = pd.read_csv(trainpath)
    trainNumAll = [30000, 30000, 30000, 30000]
    testNumAll = [1900, 1900, 1900, 1900]
    cnttrain = [0, 0, 0, 0]
    traindata = []
    for i in range(len(df)):
        traindata.append([int(df['Class Index'][i]) - 1, df['Title'][i] + ' ' + df['Description'][i], int(0)])
    traindata = np.array(traindata)
    swl = random.sample(range(4), 2)
    alllabeldatatrain = []
    for i in range(120000):
        lb = int(traindata[i][0])
        contxt = traindata[i][1]
        flag = traindata[i][2]
        if lb == swl[0] and cnttrain[lb] < trainNumAll[lb] * ratio:
            newlabel = swl[1]
            alllabeldatatrain.append([newlabel, contxt, 1])
            cnttrain[lb] += 1
        elif lb == swl[1] and cnttrain[lb] < trainNumAll[lb] * ratio:
            newlabel = swl[0]
            alllabeldatatrain.append([newlabel, contxt, 1])
            cnttrain[lb] += 1
        else:
            alllabeldatatrain.append([lb, contxt, 0])
    alllabeldatatrain = np.array(alllabeldatatrain)
    np.save(savepath + 'ranlabeltraindata', alllabeldatatrain)
    print("train data shape: ", alllabeldatatrain.shape)

    cnttest = [0, 0, 0, 0]
    df2 = pd.read_csv(testpath)
    testdata = []
    for i in range(len(df2)):
        testdata.append([int(df2['Class Index'][i]) - 1, df2['Title'][i] + ' ' + df2['Description'][i], int(0)])
    testdata = np.array(testdata)

    alllabeldatatest = []
    for i in range(7600):
        lb = int(testdata[i][0])
        contxt = testdata[i][1]
        flag = testdata[i][2]
        if lb == swl[0] and cnttest[lb] < testNumAll[lb] * ratio:
            newlabel = swl[1]
            alllabeldatatest.append([newlabel, contxt, 1])
            cnttest[lb] += 1
        elif lb == swl[1] and cnttest[lb] < testNumAll[lb] * ratio:
            newlabel = swl[0]
            alllabeldatatest.append([newlabel, contxt, 1])
            cnttest[lb] += 1
        else:
            alllabeldatatest.append([lb, contxt, 0])
    alllabeldatatest = np.array(alllabeldatatest)
    np.save(savepath + 'ranlabeltestdata', alllabeldatatest)
    print("test data shape: ", alllabeldatatest.shape)


def save_alldirty_data(trainpath, testpath, savepath, ratio):
    df = pd.read_csv(trainpath)
    dt = pd.read_csv(args.noisedatapath)
    print(dt.head())
    dtdata = []
    for i in range(len(dt)):
        dtdata.append([dt['review'][i]])
    dtdata = np.array(dtdata)
    trainNumAll = [30000, 30000, 30000, 30000]
    testNumAll = [1900, 1900, 1900, 1900]
    cnttrain = [0, 0, 0, 0]
    traindata = []
    for i in range(len(df)):
        traindata.append([int(df['Class Index'][i]) - 1, df['Title'][i] + ' ' + df['Description'][i], int(0)])
    traindata = np.array(traindata)

    alllabeldatatrain = []
    for i in range(120000):
        lb = int(traindata[i][0])
        contxt = traindata[i][1]
        flag = traindata[i][2]
        if cnttrain[lb] < trainNumAll[lb] * ratio:
            ind = random.randint(0, 90000)
            contxt = dtdata[ind][0]
            alllabeldatatrain.append([lb, contxt[:300], 1])
            cnttrain[lb] += 1
        else:
            alllabeldatatrain.append([lb, contxt, 0])
        print(i)

    alllabeldatatrain = np.array(alllabeldatatrain)
    np.save(savepath + 'alldirtytraindata', alllabeldatatrain)
    print("train data shape: ", alllabeldatatrain.shape)
    print(cnttrain)

    cnttest = [0, 0, 0, 0]
    df2 = pd.read_csv(testpath)
    testdata = []
    for i in range(len(df2)):
        testdata.append([int(df2['Class Index'][i]) - 1, df2['Title'][i] + ' ' + df2['Description'][i], int(0)])
    testdata = np.array(testdata)

    alllabeldatatest = []
    for i in range(7600):
        lb = int(testdata[i][0])
        contxt = testdata[i][1]
        flag = testdata[i][2]
        if cnttest[lb] < testNumAll[lb] * ratio:
            ind=random.randint(0,90000)
            contxt=dtdata[ind][0]
            alllabeldatatest.append([lb, contxt[:300], 1])
            cnttest[lb] += 1
        else:
            alllabeldatatest.append([lb, contxt, 0])
        print(i)
    alllabeldatatest = np.array(alllabeldatatest)
    np.save(savepath + 'alldirtytestdata', alllabeldatatest)
    print("test data shape: ", alllabeldatatest.shape)
    print(cnttest)


def save_randirty_data(trainpath, testpath, savepath, ratio):
    df = pd.read_csv(trainpath)
    dt = pd.read_csv(args.noisedatapath)
    print(dt.head())
    dtdata = []
    for i in range(len(dt)):
        dtdata.append([dt['review'][i]])
    dtdata = np.array(dtdata)
    trainNumAll = [30000, 30000, 30000, 30000]
    testNumAll = [1900, 1900, 1900, 1900]
    cnttrain = [0, 0, 0, 0]
    traindata = []
    for i in range(len(df)):
        traindata.append([int(df['Class Index'][i]) - 1, df['Title'][i] + ' ' + df['Description'][i], int(0)])
    traindata = np.array(traindata)

    swl = random.sample(range(4), 1)
    alllabeldatatrain = []
    for i in range(120000):
        lb = int(traindata[i][0])
        contxt = traindata[i][1]
        flag = traindata[i][2]
        if lb==swl[0] and cnttrain[lb] < trainNumAll[lb] * ratio:
            ind = random.randint(0, 90000)
            contxt = dtdata[ind][0]
            alllabeldatatrain.append([lb, contxt[:300], 1])
            cnttrain[lb] += 1
        else:
            alllabeldatatrain.append([lb, contxt, 0])
        print(i)

    alllabeldatatrain = np.array(alllabeldatatrain)
    np.save(savepath + 'randirtytraindata', alllabeldatatrain)
    print("train data shape: ", alllabeldatatrain.shape)
    print(cnttrain)

    cnttest = [0, 0, 0, 0]
    df2 = pd.read_csv(testpath)
    testdata = []
    for i in range(len(df2)):
        testdata.append([int(df2['Class Index'][i]) - 1, df2['Title'][i] + ' ' + df2['Description'][i], int(0)])
    testdata = np.array(testdata)

    alllabeldatatest = []
    for i in range(7600):
        lb = int(testdata[i][0])
        contxt = testdata[i][1]
        flag = testdata[i][2]
        if lb==swl[0] and cnttest[lb] < testNumAll[lb] * ratio:
            ind=random.randint(0,90000)
            contxt=dtdata[ind][0]
            alllabeldatatest.append([lb, contxt[:300], 1])
            cnttest[lb] += 1
        else:
            alllabeldatatest.append([lb, contxt, 0])
        print(i)
    alllabeldatatest = np.array(alllabeldatatest)
    np.save(savepath + 'randirtytestdata', alllabeldatatest)
    print("test data shape: ", alllabeldatatest.shape)
    print(cnttest)

if __name__ == "__main__":
    if args.datatype == 'original':
        save_org_data(trainpath=args.trainpath, testpath=args.testpath,savepath=args.savepath)
    if args.datatype == 'RandomLabelNoise':
        save_alllabel_data(trainpath=args.trainpath, testpath=args.testpath,savepath=args.savepath,ratio=args.ratio)
    if args.datatype == 'SpecificLabelNoise':
        save_ranlabel_data(trainpath=args.trainpath, testpath=args.testpath,savepath=args.savepath,ratio=args.ratio)
    if args.datatype == 'RandomDataNoise':
        save_alldirty_data(trainpath=args.trainpath, testpath=args.testpath,savepath=args.savepath,ratio=args.ratio)
    if args.datatype == 'SpecificDataNoise':
        save_randirty_data(trainpath=args.trainpath, testpath=args.testpath,savepath=args.savepath,ratio=args.ratio)