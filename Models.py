import torch
import torch.optim as optim
import torchvision 
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import pprint as pp
import numpy as np
import PIL

DEBUG = False

class Encoder(nn.Module):

    def __init__(self, encoding_channels, dim):
        super().__init__()
        if DEBUG == True:
            print(self)
        self.encoding_channels = encoding_channels
        self.dim = dim # keeps track of tensor WxH as it is passed down the network

        if DEBUG == True:
            print(self.dim)

        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size= 5, stride= 1, padding= 2)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.pool1 = torch.nn.MaxPool2d(kernel_size= 2)
        self.dim = self.dim // 2
        if DEBUG == True:
            print(self.dim)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size= 5, stride= 1, padding= 2)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.pool2 = torch.nn.MaxPool2d(kernel_size= 2)
        self.dim = self.dim // 2
        if DEBUG == True:
            print(self.dim)
        self.conv3 = torch.nn.Conv2d(16, self.encoding_channels, kernel_size= 5, stride= 1, padding= 2)

    def forward(self, x):

        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x))
        x = self.pool2(x)
        x = F.sigmoid(self.conv3(x))

        return x

class Decoder(nn.Module):
    def __init__(self, encoding_channels):

        super().__init__()
        self.encoding_channels = encoding_channels
        if DEBUG == True:
            print(self)

        self.conv1 = torch.nn.ConvTranspose2d(self.encoding_channels, 16, kernel_size= 5, stride= 1, padding= 2)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.ConvTranspose2d(16, 8, kernel_size= 5, stride= 1, padding= 2)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.ConvTranspose2d(8, 3, kernel_size= 5, stride= 1, padding= 2)
        self.conv4 = torch.nn.Conv2d(3, 3, kernel_size= 1, stride= 1, padding= 0)

    def forward(self, x):

        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x))
        x = F.sigmoid(self.conv3(x))
        x = self.conv4(x)

        return x

class Classifier(nn.Module):
    def __init__(self, encoding_channels, dim):

        super().__init__()
        self.encoding_channels = encoding_channels
        self.dim = dim
        if DEBUG == True:
            print(self)
            print(self.dim)

        self.conv1 = torch.nn.Conv2d(self.encoding_channels, 16, kernel_size = 3, stride = 1, padding= 1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.pool1 = torch.nn.MaxPool2d(kernel_size = 2)
        self.dim = self.dim // 2
        if DEBUG == True:
            print(self.dim)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding= 1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.pool2 = torch.nn.MaxPool2d(kernel_size = 2)
        self.dim = self.dim // 2
        if DEBUG == True:
            print(self.dim)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding= 1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.pool3 = torch.nn.MaxPool2d(kernel_size = 2)
        self.dim = self.dim // 2
        if DEBUG == True:
            print(self.dim)
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding= 1)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.pool4 = torch.nn.MaxPool2d(kernel_size = 2)
        self.dim = self.dim // 2
        if DEBUG == True:
            print(self.dim)
        self.fc1 = torch.nn.Linear(self.dim*self.dim*128, 512)
        self.fc2 = torch.nn.Linear(512, 47)

        self.dim = dim
    
    def forward(self, x):

        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.leaky_relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.leaky_relu(self.bn4(x))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        return x

    
class Model():
    def __init__(self, train_size, encoding_channels = 3):

        self.encoder = Encoder(encoding_channels = encoding_channels, dim = train_size)
        self.decoder = Decoder(encoding_channels = encoding_channels)
        self.classif = Classifier(encoding_channels = encoding_channels, dim = self.encoder.dim)
        self.encoding_channels = encoding_channels

    def dream(self, img1, img2):

        enc1 = self.encoder.forward(img1)
        enc2 = self.encoder.forward(img2)

        squash = (enc1 + enc2) / 2

        img_out =  Decoder(squash)

        return img_out