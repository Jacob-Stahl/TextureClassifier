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
from math import ceil

DEBUG = False

class Res_encoder(nn.Module):
    def __init__(self, encoding_channels, dim = 256):
        super().__init__()

        self.encoding_channels = encoding_channels
        self.dim = dim # keeps track of tensor WxH as it is passed down the network

        if DEBUG == True:
            print(self.dim)

        self.conv1 = nn.Conv2d(3, 8, kernel_size = 3, padding = 1)

        self.conv1a = nn.Conv2d(8, 8, kernel_size = 3, padding = 1)
        self.conv1b = nn.Conv2d(8, 8, kernel_size = 3, padding = 1)
        self.conv1c = nn.Conv2d(8, 8, kernel_size = 3, padding = 1)

        self.bn1  = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2)
        self.dim = self.dim // 2
        if DEBUG == True:
            print(self.dim)
        self.conv2 =  nn.Conv2d(8, 16, kernel_size = 3, padding = 1)

        self.conv2a =  nn.Conv2d(16, 16, kernel_size = 3, padding = 1)
        self.conv2b =  nn.Conv2d(16, 16, kernel_size = 3, padding = 1)
        self.conv2c =  nn.Conv2d(16, 16, kernel_size = 3, padding = 1)

        self.bn2  = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)
        self.dim = self.dim // 2
        if DEBUG == True:
            print(self.dim)
        self.conv3 =  nn.Conv2d(16, 32, kernel_size = 3, padding = 1)

        self.conv3a =  nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
        self.conv3b =  nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
        self.conv3c =  nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
        
        self.bn3  = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2)
        self.dim = self.dim // 2
        if DEBUG == True:
            print(self.dim)
        self.conv4 =  nn.Conv2d(32, 64, kernel_size = 3, padding = 1)

        self.conv4a =  nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.conv4b =  nn.Conv2d(64, 64, kernel_size = 3, padding = 1)

        self.bn4  = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, self.encoding_channels, kernel_size = 5, padding = 2)      


    def forward(self, x):

        x = self.conv1(x)
        res = x
        x = F.leaky_relu(x)
        x = F.leaky_relu(self.conv1a(x))
        x = F.leaky_relu(self.conv1b(x))
        x = F.leaky_relu(self.conv1c(x) + res)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        res = x
        x = F.leaky_relu(x)
        x = F.leaky_relu(self.conv2a(x))
        x = F.leaky_relu(self.conv2b(x))
        x = F.leaky_relu(self.conv2c(x) + res)
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        res = x
        x = F.leaky_relu(x)
        x = F.leaky_relu(self.conv3a(x))
        x = F.leaky_relu(self.conv3b(x))
        x = F.leaky_relu(self.conv3c(x) + res)
        x = self.bn3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        res = x
        x = F.leaky_relu(x)
        x = F.leaky_relu(self.conv4a(x))
        x = F.leaky_relu(self.conv4b(x) + res)
        x = self.bn4(x)
        x = F.leaky_relu(self.conv5(x))

        return x

class Res_decoder(nn.Module):
    def __init__(self, encoding_channels, dim):
        self.encoding_channels = encoding_channels
        self.dim = dim
        if DEBUG == True:
            print(self)
            print(self.dim)

        self.conv1 = torch.nn.ConvTranspose2d(self.encoding_channels, 64, kernel_size = 3, padding= 1)

        self.conv1a = torch.nn.ConvTranspose2d(64, 64, kernel_size = 3, padding= 1)
        self.conv1b = torch.nn.ConvTranspose2d(64, 64, kernel_size = 3, padding= 1)
        self.conv1c = torch.nn.ConvTranspose2d(64, 64, kernel_size = 3, padding= 1)

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.ConvTranspose2d(64, 32, stride = 2, kernel_size = 3, padding= 1)

        self.conv2a = torch.nn.ConvTranspose2d(32, 32, kernel_size = 3, padding = 1)
        self.conv2b = torch.nn.ConvTranspose2d(32, 32, kernel_size = 3, padding = 1)
        self.conv2c = torch.nn.ConvTranspose2d(32, 32, kernel_size = 3, padding = 1)

        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.ConvTranspose2d(32, 16, stride = 2, kernel_size = 3, padding= 1)

        self.conv3a = torch.nn.ConvTranspose2d(16, 16, kernel_size = 3, padding= 1)
        self.conv3b = torch.nn.ConvTranspose2d(16, 16, kernel_size = 3, padding= 1)
        self.conv3c = torch.nn.ConvTranspose2d(16, 16, kernel_size = 3, padding= 1)

        self.bn3 = torch.nn.BatchNorm2d(16)
        self.conv4 = torch.nn.ConvTranspose2d(16, 8, stride = 2, kernel_size = 3, padding= 1)

        self.conv4a = torch.nn.ConvTranspose2d(8, 8, kernel_size = 3, padding= 1)
        self.conv4b = torch.nn.ConvTranspose2d(8, 8, kernel_size = 3, padding= 1)
        self.conv4c = torch.nn.ConvTranspose2d(8, 8, kernel_size = 3, padding= 1)

        self.conv5 = torch.nn.ConvTranspose2d(8, 3, kernel_size = 3, padding = 1)
    
    def forward(self, x):

        x = self.conv1(x))
        res = x
        x = F.leaky_relu(x)
        x = F.leaky_relu(self.conv1a(x))
        x = F.leaky_relu(self.conv1b(x))
        x = F.leaky_relu(self.conv1c(x) + res)
        x = self.bn1(x)

        x = self.conv2(x)
        res = x
        x = F.leaky_relu(x)
        x = F.leaky_relu(self.conv2a(x))
        x = F.leaky_relu(self.conv2b(x))
        x = F.leaky_relu(self.conv2c(x) + res)
        x = self.bn2(x)

        x = self.conv3(x)
        res = x
        x = F.leaky_relu(x)
        x = F.leaky_relu(self.conv3a(x))
        x = F.leaky_relu(self.conv3b(x))
        x = F.leaky_relu(self.conv3c(x) + res)
        x = self.bn3(x)

        x = self.conv4(x)
        res = x
        x = F.sigmoid(x)
        x = F.sigmoid(self.conv4a(x))
        x = F.sigmoid(self.conv4b(x))
        x = F.sigmoid(self.conv4c(x) + res)

        x = self.conv5(x)

        return x

class Conv_classifier:
    def __init__(self, encoding_channels, dim):
        self.encoding_channels = encoding_channels
        self.dim = dim

        self.conv1 = torch.nn.Conv2d(self.encoding_channels, 16, kernel_size = 5, padding= 2)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.dim = self.dim // 2
        if DEBUG == True:
            print(self.dim)

        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size = 5, padding= 2)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.dim = self.dim // 2
        if DEBUG == True:
            print(self.dim)
        
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size = 5, padding= 2)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.pool3 = torch.nn.MaxPool2d(2)
        self.dim = self.dim // 2
        if DEBUG == True:
            print(self.dim)

        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size = 5, padding= 2)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.pool4 = torch.nn.MaxPool2d(2)
        self.dim = self.dim // 2
        if DEBUG == True:
            print(self.dim)

        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size = 5, padding= 2)
        self.bn5 = torch.nn.BatchNorm2d(256)
        self.pool5 = torch.nn.MaxPool2d(2)
        self.dim = self.dim // 2
        if DEBUG == True:
            print(self.dim)

        self.conv6 = torch.nn.Conv2d(256, 47, kernel_size = 1, padding = 0)

    def forward(self, x):

        x = F.leaky_relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        
        x = F.leaky_relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        x = F.leaky_relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)

        x = F.leaky_relu(self.conv4(x))
        x = self.bn4(x)
        x = self.pool4(x)

        x = F.leaky_relu(self.conv5(x))
        x = self.bn5(x)
        x = self.pool5(x)

        x = F.sigmoid(self.conv6(x))

        return torch.squeeze(x)

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
    def __init__(self, encoding_channels, dim):

        super().__init__()
        self.encoding_channels = encoding_channels
        self.dim = dim
        if DEBUG == True:
            print(self)
            print(self.dim)

        self.conv1 = torch.nn.ConvTranspose2d(self.encoding_channels, 32, kernel_size= 5, stride= 2, padding= 2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.dim = 2 * (self.dim - 1) + 5 - 2 * 2
        if DEBUG == True:
            print(self.dim)
        self.conv2 = torch.nn.ConvTranspose2d(32, 16, kernel_size= 5, stride= 2, padding= 2)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.dim = 2 * (self.dim - 1) + 5 - 2 * 2
        if DEBUG == True:
            print(self.dim)
        self.conv3 = torch.nn.ConvTranspose2d(16, 8, kernel_size= 6, stride= 1, padding= 2)
        self.dim = 1 * (self.dim - 1) + 6 - 2 * 1
        if DEBUG == True:
            print(self.dim)
        self.conv4 = torch.nn.Conv2d(8, 3, kernel_size= 1, stride= 1, padding= 1)
        if DEBUG == True:
            print(self.dim)

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
    def __init__(self, train_size, encoding_channels = 10):

        self.encoder = Encoder(encoding_channels = encoding_channels, dim = train_size)
        self.decoder = Decoder(encoding_channels = encoding_channels, dim = self.encoder.dim)
        self.classif = Classifier(encoding_channels = encoding_channels, dim = self.encoder.dim)
        self.encoding_channels = encoding_channels

    def simple_dream(self, img1):

        enc = self.encoder.forward(img1)
        img_out =  self.decoder(enc)

        return img_out

    def noise_injection(self, img):

        enc = self.encoder(img)
        noise = (torch.rand(enc.size()) - .5) / 10
        noise = noise.to(enc.device)
        out_img = self.decoder(enc + noise)

        return out_img

    def cross_dream(self, img1, img2): # images must be the same size

        enc1 = self.encoder(img1)
        enc2 = self.encoder(img2)
        mg_out = self.decoder(enc1 * enc2)

        return img_out

    def fibonacci_dream(self, img1):

        temp1 = img1
        temp2 = self.simple_dream(img1)
        temp3 = None

        for i in range(3):

            temp3 = self.cross_dream(temp1, temp2)
            temp1 = temp2
            temp2 = temp3
            temp3 = None

        return temp2