
import numpy as np
import torch

from torch import nn

from torchvision import transforms

from train_model.models import LeNet1, LeNet5

def PreLoss(model, predatapath, ratio, newdatasavepath):

    predata = np.load(predatapath, allow_pickle=True)
    x_train = torch.from_numpy(np.array([x / 255. for x in predata[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in predata[:, 0]]))
    is_dirty = torch.from_numpy(np.array([int(x) for x in predata[:, 2]]))
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
    ])
    x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))
    loss_fn = nn.CrossEntropyLoss()
    ranklist = [[], [], [], [], [], [], [], [], [], []]
    with torch.no_grad():
        for i in range(60000):
            out = model(x_train[i].reshape(1, 1, 28, 28))
            y = []
            y.append(y_train[i])
            y = np.array(y)
            y = torch.from_numpy(y)
            y = y.long()
            tt, pred = torch.max(out, axis=1)
            cur_loss = loss_fn(out, y)
            ranklist[int(predata[i][0])].append([predata[i][0], predata[i][1], predata[i][2], float(cur_loss)])
    newdata = []
    for i in range(10):
        tmp = []
        for j in range(len(ranklist[i])):
            tmp.append(ranklist[i][j])
        tmp = np.array(tmp, dtype=object)
        tmp = tmp[tmp[:, 3].argsort()[::-1]]

        tmp = tmp[int(tmp.shape[0] * ratio):]

        for j in range(tmp.shape[0]):
            newdata.append(tmp[j])

    np.save(newdatasavepath, newdata)
if __name__ == "__main__":
    datapath = './data/MNIST/MNIST_PNG/alllabeltraindata.npy'
    modelpath = './models/mnist_alllabel_LeNet5.pth'
    savedatapath = './data/MNIST/MNIST_PNG/alllabeltraindata_PreLoss.npy'

    model = LeNet5()
    state_dict = torch.load(modelpath)
    model.load_state_dict(state_dict)
    PreLoss(model, datapath, 0.05, savedatapath)