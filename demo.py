import logging
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from pyod.models.vae import VAE
from torchvision import transforms
from sklearn.cluster import KMeans
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s -  %(message)s')
logger = logging.getLogger(__name__)
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

        #Outlier
        x_train_VAE = x_train.reshape(x_train.shape[0], 28 * 28)
        clf = VAE(epochs=5,verbose=0)
        clf.fit(x_train_VAE)
        y_score = clf.decision_scores_
        y_score = np.array(y_score)

        #Activation and PreLoss





        res_Outlier = []
        for i in range(tmp.shape[0]):
            res_Outlier.append([tmp[i][0], tmp[i][1], tmp[i][2], y_score[i]])


    newdata = np.array(newdata,dtype=object)
    return newdata


