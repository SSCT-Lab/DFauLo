import torch
from torch import nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 16, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 32, 3, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 3, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet20():
    return ResNet(ResidualBlock)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):

    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )
        #         self.classifier = nn.Linear(512,10)

        self._initialize_weight()

    def forward(self, x):
        out = self.features(x)
        # 在进入
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    # make layers
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3  # RGB 初始通道为3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # kernel_size 为 2 x 2,然后步长为2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),  # 都是(3.3)的卷积核
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]  # RelU
                in_channels = x  # 重定义通道
        #         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    # 初始化参数
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # xavier is used in VGG's paper
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
class LSTM(nn.Module):
    def __init__(self, voc_len, PAD):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=voc_len, embedding_dim=200, padding_idx=PAD)
        self.lstm = nn.LSTM(input_size=200, hidden_size=128, num_layers=2,batch_first=True,dropout=0.5)

        self.fc1 = nn.Linear(128, 64)
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(64, 4)
        # self.relu = nn.ReLU()

    def forward(self, input):
        embd = self.embedding(input)
        output, (h_n, c_n) = self.lstm(embd)
        # out = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)
        out = self.fc1(output[:, -1, :])
        out=self.relu(out)
        out=self.fc2(out)

        return out
class BiLSTM(nn.Module):
    def __init__(self, voc_len, PAD):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=voc_len, embedding_dim=200, padding_idx=PAD)
        self.lstm = nn.LSTM(input_size=200, hidden_size=128, num_layers=2,bidirectional=True,batch_first=True,dropout=0.5)
        self.fc1 = nn.Linear(128*2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)
        # self.fc2 = nn.Linear(64, 4)
        # self.relu = nn.ReLU()

    def forward(self, input):
        embd = self.embedding(input)
        output, (h_n, c_n) = self.lstm(embd)
        # out = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)
        out = self.fc1(output[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)

        return out
class TCDCNN(nn.Module):

    def __init__(self):
        super(TCDCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 48, kernel_size=3)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=2)

        self.linear_1 = nn.Linear(256, 10)
        # self.linear_2 = nn.Linear(256, 2)
        # self.linear_3 = nn.Linear(256, 2)
        # self.linear_4 = nn.Linear(256, 2)
        # self.linear_5 = nn.Linear(256, 5)
        self.dropout = nn.Dropout()

    def loss(self, pred, y):
        criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()

        # landmark = y[:,0:10].float()

        # gender = gender.view(-1,gender.size()[0])

        # smile = y[:,12:14].view(1,2).long()
        # glass = y[:,14:16].view(1,2).long()
        # pose =  y[:,16:21].view(1,5).long()
        loss_mse = mse_criterion(pred[0], y[0])
        # loss_gender = criterion(pred[1], y[1])
        # loss_smile = criterion(pred[2], y[2])
        # loss_glass = criterion(pred[3], y[3])
        # loss_pose = criterion(pred[4], y[4])

        loss = loss_mse
        return loss

    def classifier(self, x):
        x_one = self.linear_1(x)
        # x_two = F.softmax(self.linear_2(x))
        # x_three = F.softmax(self.linear_3(x))
        # x_four = F.softmax(self.linear_4(x))
        # x_five = F.softmax(self.linear_5(x))
        return x_one

    def features(self, x):
        x = F.max_pool2d(F.hardtanh(self.conv1(x)), 2)
        x = F.max_pool2d(F.hardtanh(self.conv2(x)), 2)
        x = F.max_pool2d(F.hardtanh(self.conv3(x)), 2)
        x_tanh = self.conv4(x)
        x = F.hardtanh(x_tanh)

        x = x.view(-1, 256)
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x_one = self.classifier(x)
        return x_one

    def predict(self, x):
        x = self.features(x)
        x_one = self.classifier(x)
        return [x_one]

    def accuracy(self, x, y):
        landmarkeye = y[:, [0, 5]].float()
        landmarkeyey = y[:, [1, 6]].float()
        landmark = y[:, 0:10].float()

        mse_criterion = nn.MSELoss()
        loss_mse = mse_criterion(x, landmark)
        loss_base = mse_criterion(landmarkeye, landmarkeyey)
        accuracy = torch.div(loss_mse.data, loss_base.data)
        # accuracy = torch.div(accuracytemp, 0.01)
        error = accuracy.cpu().numpy()
        error_text = "error rate(%):{}".format(float(error))
        # logging.info(error_text)

        #        print("accuracy: ", accuracy.numpy(),"%")
        #        print("base: ", loss_base)
        #        logging.info("base")
        #        logging.info(float(loss_base.data.numpy()))
        return accuracy.item()
class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        self.c1 = nn.Conv2d(1, 4, 5)
        self.TANH = nn.Tanh()
        self.s2 = nn.AvgPool2d(2)
        self.c3 = nn.Conv2d(4, 12, 5)
        self.s4 = nn.AvgPool2d(2)
        self.fc = nn.Linear(12 * 4 * 4, 10)

    def forward(self, x):
        x = self.c1(x)
        x = self.TANH(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.TANH(x)
        x = self.s4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.Sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)

    def forward(self, x):
        # x输入为32*32*1， 输出为28*28*6
        x = self.Sigmoid(self.c1(x))
        # x输入为28*28*6， 输出为14*14*6
        x = self.s2(x)
        # x输入为14*14*6， 输出为10*10*16
        x = self.Sigmoid(self.c3(x))
        # x输入为10*10*16， 输出为5*5*16
        x = self.s4(x)
        # x输入为5*5*16， 输出为1*1*120
        x = self.c5(x)
        x = self.flatten(x)
        # x输入为120， 输出为84
        x = self.f6(x)
        # x输入为84， 输出为10
        x = self.f7(x)
        return x






