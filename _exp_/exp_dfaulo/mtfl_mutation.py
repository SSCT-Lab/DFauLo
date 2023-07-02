import time

import torch.autograd as A
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from tqdm import tqdm

from _exp_.train_model.tcdcnn_dataset import FaceLandmarksDataset

from _exp_.train_model.models import TCDCNN


def getlockmodelTCDCNN(modelpath):
    premodel = TCDCNN()

    state_dict = torch.load(modelpath)
    premodel.load_state_dict(state_dict)
    for param in premodel.parameters():
        param.requires_grad = False

    premodel.linear_1 = nn.Linear(256, 10)

    return premodel


def retrainTCDCN(traindatapath, trainbbxpath, modelsavepath, net):
    optim1 = optim.SGD(net.parameters(), 0.003)
    # input = Variable(torch.randn(1, 1, 32, 32))

    count = 0
    accuracy = 0
    loss = 0
    mypath = "..\\MTFL"
    train = traindatapath
    annotation = trainbbxpath
    # test = "testing_D1.txt"
    # annotation_test = "annotation_test_D1.txt"
    faceLandMark = FaceLandmarksDataset(mypath, train, annotation)
    dataloader = DataLoader(faceLandMark, batch_size=64,
                            shuffle=True, num_workers=0)
    # faceLandMark_test = FaceLandmarksDataset(mypath, test, annotation_test)
    # dataloader_test = DataLoader(faceLandMark_test, batch_size=64,
    #                              shuffle=True, num_workers=0)
    batch_size = 64

    print("Starting training")
    loss_total = 0

    epochs = 5

    final_accuracy = 0
    final_loss = 0
    final_training_accuracy = 0
    loss_list = []
    err_list = []
    epo_list = []
    errtrain_list = []
    for e in range(0, epochs):
        epo_list.append(e)
        total_accuracy = 0
        loss_total = 0
        count = 0
        total_accuracy_training = 0
        for i, data in tqdm(enumerate(dataloader, 1)):
            images, landmark, gender, smile, glass, pose = data

            # images = images.squeeze(1)
            landmark = A.Variable(landmark)
            # images = A.Variable(images)
            # gender = A.Variable(gender)
            # smile = A.Variable(smile)
            # glass = A.Variable(glass)
            # pose = A.Variable(pose)
            images_temp = images.clone()
            optim1.zero_grad()
            x_one = net(images.float())
            loss = net.loss([x_one],
                            [landmark.float()])
            loss.backward()
            optim1.step()
            # loss for training
            loss_total = loss_total + loss.item()
            accuracy_training = net.accuracy(x_one, landmark.float())
            count = count + 1
            loss_text = "Loss:{}".format(float(loss_total / count))
            # logging.info(loss_text)
            # Accuracy for testing
            net.eval()
            # accuracy = test_model(net, dataloader_test)
            total_accuracy = total_accuracy + accuracy
            # Accuracy for training
            total_accuracy_training = total_accuracy_training + accuracy_training
            net.train()
            # progress(str(e),str(loss[0].data.numpy()[0]),str(accuracy),i,int(len(dataloader.dataset)/batch_size))
        # tottestmodel(net,dataloader_test)
    #     print("loss: ", loss_total / count)
    #     print("test MNE rate: ", total_accuracy / count)
    #     print("train MNE ratio:", total_accuracy_training / count)
    #     loss_list.append(loss_total / count)
    #     err_list.append(total_accuracy / count)
    #     errtrain_list.append(total_accuracy_training / count)
    #     final_loss = final_loss + loss_total / count
    #     final_accuracy = final_accuracy + total_accuracy / count
    #     final_training_accuracy = final_training_accuracy + total_accuracy_training / count
    # torch.save(net.state_dict(), modelsavepath)
    # plt.plot(epo_list, loss_list)
    # plt.title('train loss')
    # plt.show()
    # plt.plot(epo_list, err_list, label='test MNE')
    # plt.plot(epo_list, errtrain_list, label='train MNE')
    # plt.title('MNE')
    # plt.show()
    # print("Final loss:", final_loss / epochs)
    # print("Final training error rate: ", final_training_accuracy / epochs)
    # print("Final testing Error rate: ", final_accuracy / epochs)


if __name__ == "__main__":

    traintype = 'direct'

    traindatapath = "./TCDCN_NEWDATA_OUT/training_org_VAE_MID_OUT_new.txt"
    trainbbxpath = "./TCDCN_NEWDATA_OUT/annotation_org_VAE_MID_OUT_new.txt"
    modelpath = '../models/org_tcdcn.pth'

    newmodelsavepath = '../retrainmodels/org_tcdcn_retrain_OUT_new.pth'

    # 修改dataset的剔除值！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

    if traintype == 'direct':
        model = TCDCNN()
        state_dict = torch.load(modelpath)
        model.load_state_dict(state_dict)
        start=time.time()
        retrainTCDCN(traindatapath, trainbbxpath, newmodelsavepath, model)
        end=time.time()
        print('执行时间:',end-start)

    elif traintype == 'randomweight':
        model = getlockmodelTCDCNN(modelpath)
        retrainTCDCN(traindatapath, trainbbxpath, newmodelsavepath, model)
