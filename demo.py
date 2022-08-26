import logging

import numpy as np
import torch
from matplotlib import pyplot as plt
from pyod.models.vae import VAE
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.cluster import KMeans
from tqdm import tqdm
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from train_model.models import LeNet5

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s -  %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def OAP(datapath, ratio, model):
    dataprg = np.load(datapath, allow_pickle=True)
    newdata = []
    logger.info('running Outlier Activation and PreLoss:')
    for lb in tqdm(range(10)):
        tmp = []
        for i in range(60000):
            if int(dataprg[i][0]) == lb:
                tmp.append(dataprg[i])
        tmp = np.array(tmp)
        x_train = torch.from_numpy(np.array([x / 255. for x in tmp[:, 1]]))

        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
        ])
        x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))
        y_train = torch.from_numpy(np.array([int(x) for x in tmp[:, 0]]))
        dataset = TensorDataset(x_train, y_train)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=False)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        # Outlier
        x_train_VAE = x_train.reshape(x_train.shape[0], 28 * 28)
        clf_vae = VAE(epochs=5, verbose=0)
        clf_vae.fit(x_train_VAE)
        vae_score = clf_vae.decision_scores_
        vae_score = np.array(vae_score)

        # Activation and PreLoss
        model.eval()
        model.to(device)
        Act = []
        PreLoss = []
        with torch.no_grad():
            for batch, (X, y) in enumerate(data_loader):
                y = y.long()
                X, y = X.to(device), y.to(device)
                out = model.c1(X)
                out = model.Sigmoid(out)
                out = model.s2(out)
                out = model.c3(out)
                out = model.Sigmoid(out)
                out = model.s4(out)
                out = model.c5(out)
                out = model.flatten(out)
                Act_out = out.cpu().numpy()
                out = model.f6(out)
                out = model.f7(out)
                Loss = loss_fn(out, y).cpu().numpy()
                for i in range(Act_out.shape[0]):
                    Act.append(Act_out[i])
                    PreLoss.append(Loss[i])
        Act = np.array(Act)
        PreLoss = np.array(PreLoss)

        clf_kmeans = KMeans(n_clusters=2)
        clf_kmeans.fit(Act)
        km_label = clf_kmeans.labels_

        label1num = 0
        label0num = 0
        for i in range(km_label.shape[0]):
            if km_label[i] == 1:
                label1num += 1
            elif km_label[i] == 0:
                label0num += 1

        seldata = -1
        if label1num <= label0num:
            seldata = 0
        else:
            seldata = 1

        res = []
        for i in range(tmp.shape[0]):
            if km_label[i] == seldata:
                res.append([tmp[i][0], tmp[i][1], tmp[i][2], vae_score[i], 0, PreLoss[i]])
            else:
                res.append([tmp[i][0], tmp[i][1], tmp[i][2], vae_score[i], 1, PreLoss[i]])

        res = np.array(res, dtype=object)
        res = res[res[:, 3].argsort()[::-1]]  # vae score sort
        for i in range(res.shape[0]):
            if i <= int(res.shape[0] * ratio):
                res[i, 3] = 1
            else:
                res[i, 3] = 0
        res = res[res[:, 5].argsort()[::-1]]  # loss sort
        for i in range(res.shape[0]):
            if i <= int(res.shape[0] * ratio):
                res[i, 5] = 1
            else:
                res[i, 5] = 0
            newdata.append(res[i])
    logger.info('finish OAP.')
    newdata = np.array(newdata, dtype=object)
    print(newdata.shape)
    return newdata


if __name__ == "__main__":
    model = LeNet5()
    state_dict = torch.load('./demodata/mnist_RandomLabelNoise_LeNet5.pth')
    model.load_state_dict(state_dict)

    OAP(datapath='./demodata/RandomLabelNoiseData.npy',
        ratio=0.05,
        model=model)
