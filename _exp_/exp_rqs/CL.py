import random
import time

import cleanlab
import matplotlib.pyplot as plt
import numpy as np
import torch
from skorch import NeuralNetClassifier
from torch import nn
from torch.utils.data import TensorDataset
from torchvision import transforms

from _exp_.train_model.models import LeNet5, LeNet1
from sklearn.model_selection import cross_val_predict, KFold


def PoBL(ranklist, ratio):
    n = ranklist.shape[0]
    m = 0
    for i in range(n):
        if ranklist[i] == 1:
            m += 1
    cnt = 0
    for i in range(int(n * ratio)):
        if ranklist[i] == 1:
            cnt += 1
    # print('PoBL score: ', cnt / m)
    return cnt / m


def RAUC(ranklist, bestlist):
    rat = [x for x in range(101)]
    y = []
    yb = []
    for i in rat:
        y.append(PoBL(ranklist, i / 100.) * 100)
        yb.append(PoBL(bestlist, i / 100.) * 100)
    colorlist = ['violet', 'green', 'red', 'hotpink', 'mediumblue', 'orange', 'yellow', 'yellowgreen', 'peachpuff']
    # plt.plot(rat, y, color=colorlist[PCNT], label=str(PCNT + 1))
    # if PCNT==0:
    # plt.plot(rat, yb, color='blue', label='best')
    # plt.legend()
    # if PCNT==PALL-1:
    # plt.show()
    # print("RAUC score: ", np.trapz(y, rat) / np.trapz(yb, rat))
    return np.trapz(y, rat) / np.trapz(yb, rat), y, yb


def bestAUC(ranklist):
    bestlist = []
    for i in range(ranklist.shape[0]):
        if ranklist[i] == 1:
            bestlist.append(ranklist[i])
    for i in range(ranklist.shape[0]):
        if ranklist[i] == 0:
            bestlist.append(ranklist[i])
    bestlist = np.array(bestlist)
    return bestlist


def EXAM_F(ranklist):
    n = ranklist.shape[0]
    pos = -1
    for i in range(n):
        if ranklist[i] == 1:
            pos = i
            break
    return (pos + 1) / n
    # print('EXAM_F score: ', (pos + 1) / n)


def EXAM_L(ranklist):
    n = ranklist.shape[0]
    pos = -1
    for i in range(n - 1, -1, -1):
        if ranklist[i] == 1:
            pos = i
            break
    return (pos + 1) / n
    # print('EXAM_L score: ', (pos + 1) / n)


def EXAM_AVG(ranklist):
    n = ranklist.shape[0]
    m = 0
    tf = 0
    for i in range(n):
        if ranklist[i] == 1:
            tf += i
            m += 1
    return tf / (n * m)
    # print('EXAM_AVG score: ', tf / (n * m))


def APFD(ranklist):
    n = ranklist.shape[0]
    m = 0
    tf = 0
    for i in range(n):
        if ranklist[i] == 1:
            tf += i
            m += 1

    # print('APFD score: ', 1 - (tf / (n * m)) + (1 / (2 * n)))
    return 1 - (tf / (n * m)) + (1 / (2 * n))


def CLEANLAB_noCV(datatype, modeltype):
    fea = np.load('./data/MNIST/MNIST_LR_feature/' + 'CL_' + datatype + '_feature_' + modeltype + '.npy',
                  allow_pickle=True)
    # print(fea[3,0:10])
    lb = fea[:, -1].astype(int)
    is_dirty = fea[:, -3].astype(int)
    pp = fea[:, 0:10]
    print(lb.shape)
    print(pp.shape)

    ind = [x for x in range(pp.shape[0])]
    random.shuffle(ind)
    pp = pp[ind]
    lb = lb[ind]
    is_dirty = is_dirty[ind]
    # for i in range(pp.shape[0]):
    #     pp[i], pp[ind[i]] = pp[ind[i]], pp[i]
    #     lb[i], lb[ind[i]] = lb[ind[i]], lb[i]
    #     is_dirty[i], is_dirty[ind[i]] = is_dirty[ind[i]], is_dirty[i]

    Qs = cleanlab.rank.get_label_quality_scores(
        labels=lb,
        pred_probs=pp,
        method='self_confidence',
        adjust_pred_probs=False
    )

    rk = []
    for i in range(Qs.shape[0]):
        rk.append([is_dirty[i], Qs[i]])
    rk = np.array(rk)
    rk = rk[rk[:, 1].argsort()]
    rank = rk[:, 0]
    save3, y, yb = RAUC(rank, bestAUC(rank))
    f = EXAM_F(rank)
    l = EXAM_L(rank)
    avg = EXAM_AVG(rank)
    save1 = PoBL(rank, 0.1)
    save2 = APFD(rank)
    return save1, save2, save3, f, l, avg, y, yb


device = 'cuda'


def val(dataloader, model, loss_fn):
    # 将模型转为验证模式
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    # 非训练，推理期用到（测试时模型参数不用更新， 所以no_grad）
    # print(torch.no_grad)
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            y = y.long()
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print('val_loss' + str(loss / n))
        print('val_acc' + str(current / n))

        return current / n


def train_model(model, lr, epoch, train_loader, test_loader):
    min_acc = 0
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    model.to(device)
    for t in range(epoch):
        loss, current, n = 0.0, 0.0, 0
        for batch, (X, y) in enumerate(train_loader):
            y = y.long()
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print('train_loss' + str(loss / n))
        print('train_acc' + str(current / n))
        a = val(test_loader, model, loss_fn)
        print(f"epoch{t + 1} loss{a}\n-------------------")
    print("DONE")
    return model


def CLEANLAB_CV(datatype, modeltype):
    trainarr = np.load('./data/MNIST/MNIST_PNG/' + datatype + 'traindata.npy', allow_pickle=True)
    testarr = np.load('./data/MNIST/MNIST_PNG/' + datatype + 'testdata.npy', allow_pickle=True)

    x_train = torch.from_numpy(np.array([x / 255. for x in trainarr[:, 1]]))
    y_train = torch.from_numpy(np.array([int(x) for x in trainarr[:, 0]]))
    is_dirty = torch.from_numpy(np.array([int(x) for x in trainarr[:, 2]]))

    x_test = torch.from_numpy(np.array([x / 255. for x in testarr[:, 1]]))
    y_test = torch.from_numpy(np.array([int(x) for x in testarr[:, 0]]))

    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
    ])
    x_train = data_transform(x_train.reshape(x_train.shape[0], 3, 28, 28))
    x_test = data_transform(x_test.reshape(x_test.shape[0], 3, 28, 28))

    NUM = trainarr.shape[0]
    ind = [x for x in range(trainarr.shape[0])]
    random.shuffle(ind)
    x_train = x_train[ind]
    y_train = y_train[ind]
    is_dirty = is_dirty[ind]

    cross_num = 3
    X = []
    Y = []
    D = []
    for i in range(cross_num):
        X.append(x_train[int(NUM / cross_num * i):int(NUM / cross_num * (i + 1))])
        Y.append(y_train[int(NUM / cross_num * i):int(NUM / cross_num * (i + 1))])
        D.append(is_dirty[int(NUM / cross_num * i):int(NUM / cross_num * (i + 1))])
    print(len(X))
    print(len(X[0]))

    # TT = torch.tensor([])
    # TT = torch.cat((TT, X[0]), 0)
    # TT = torch.cat((TT, X[1]), 0)
    # print(TT.shape)

    sfout = []

    for i in range(cross_num):

        now_x = torch.tensor([])
        now_y = torch.tensor([])
        now_d = torch.tensor([])

        cv_x = torch.tensor([])
        cv_y = torch.tensor([])
        cv_d = torch.tensor([])

        for j in range(cross_num):
            if j != i:
                now_x = torch.cat((now_x, X[j]), 0)
                now_y = torch.cat((now_y, Y[j]), 0)
                now_d = torch.cat((now_d, D[j]), 0)
            if j == i:
                cv_x = torch.cat((now_x, X[j]), 0)
                cv_y = torch.cat((now_y, Y[j]), 0)
                cv_d = torch.cat((now_d, D[j]), 0)

        traindataset = TensorDataset(now_x, now_y)
        train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=16, shuffle=True)

        testdataset = TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=16, shuffle=True)

        model = LeNet1()
        lr = 0.001
        epoch = 85

        model_trained = train_model(model=model, lr=lr, epoch=epoch, train_loader=train_loader, test_loader=test_loader)

        model_trained.eval()

        softmax_func = nn.Softmax(dim=1)

        with torch.no_grad():
            for i in range(cv_x.shape[0]):
                tmp = cv_x[i].reshape(1, 1, 28, 28)
                tmp = tmp.to(device)
                out = model_trained(tmp)
                soft_output = softmax_func(out)
                soft_output = soft_output.cpu()
                sfout.append(soft_output.numpy()[0])

    sfout = np.array(sfout)
    lb = y_train.numpy()
    Qs = cleanlab.rank.get_label_quality_scores(
        labels=lb,
        pred_probs=sfout,
        method='self_confidence',
        adjust_pred_probs=False
    )

    rk = []
    for i in range(Qs.shape[0]):
        rk.append([is_dirty[i], Qs[i]])
    rk = np.array(rk)
    rk = rk[rk[:, 1].argsort()]
    rank = rk[:, 0]
    save3, y, yb = RAUC(rank, bestAUC(rank))
    f = EXAM_F(rank)
    l = EXAM_L(rank)
    avg = EXAM_AVG(rank)
    save1 = PoBL(rank, 0.1)
    save2 = APFD(rank)
    return save1, save2, save3, f, l, avg, y, yb
