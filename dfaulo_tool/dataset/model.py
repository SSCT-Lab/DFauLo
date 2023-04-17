import torch
from torch import nn
from torchvision.transforms import transforms

from utils.dataset import dataset

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