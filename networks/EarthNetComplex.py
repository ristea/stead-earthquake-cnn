from pathlib import Path

import torch.nn as nn
from networks.complexLayers import *


class EarthNetComplex(nn.Module):
    def __init__(self):
        super(EarthNetComplex, self).__init__()
        self.conv1 = ComplexConv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.pool1 = ComplexMaxPool2d(kernel_size=2)

        self.conv2 = ComplexConv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.pool2 = ComplexMaxPool2d(kernel_size=2)

        self.conv3 = ComplexConv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv4 = ComplexConv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool3 = ComplexMaxPool2d(kernel_size=2)

        self.conv5 = ComplexConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = ComplexConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool4 = ComplexMaxPool2d(kernel_size=2)

        self.conv7 = ComplexConv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.conv8 = ComplexConv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.pool5 = ComplexMaxPool2d(kernel_size=2)

        self.conv9 = ComplexConv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv10 = ComplexConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool6 = ComplexMaxPool2d(kernel_size=2)
        
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(in_features=8192, out_features=3)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.00001)

    @staticmethod
    def get_path():
        return str(Path(__file__).absolute())

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Residual Complex Conv Block
        x = self.conv3(x)
        x = self.relu(x)
        y = x
        x = self.conv4(x)
        x = self.relu(x)
        x = x + y
        x = self.pool3(x)
        #############################

        # Residual Complex Conv Block
        x = self.conv5(x)
        x = self.relu(x)
        y = x
        x = self.conv6(x)
        x = self.relu(x)
        x = x + y
        x = self.pool4(x)
        #############################

        # Residual Complex Conv Block
        x = self.conv7(x)
        x = self.relu(x)
        y = x
        x = self.conv8(x)
        x = self.relu(x)
        x = x + y
        x = self.pool5(x)
        #############################

        x = self.conv9(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.pool6(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x[:, 0], x[:, 1], x[:, 2]
