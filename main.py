import torch
import torch.optim as optim
import torchvision 
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import pprint as pp
import numpy as np
from set import TextureDataset

def out_size(input_size, kernal_size, stride):
    out = ((input_size - kernal_size) // stride) + 1
    print(out)
    return out

def run():

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dtd\\images\\')
    set = TextureDataset(root)

    batch_size = 16
    test_size = 500
    train_set, dev_set = random_split(set, [len(set) - test_size, test_size])

    sample = set[500][0]
    print(sample.size())

    trainset = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    testset = DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=4)

    print("dev set size:     ", len(dev_set))
    print("train set size:   ", len(train_set))
    print("image shape:      ", set.image_shape)
    print("image catagories: ", set.num_catagories)

    class Net1(nn.Module):

        def __init__(self):
            super().__init__()
            self.size = set.image_shape[0]
            self.conv1 = torch.nn.Conv2d(3, 512, kernel_size = 11, stride = 7)
            self.size = out_size(self.size, 11, 7)
            self.conv2 = torch.nn.Conv2d(512, 512, kernel_size = 5, stride = 2)
            self.size = out_size(self.size, 5, 2)
            self.conv3 = torch.nn.Conv2d(512, 1024, kernel_size = 3, stride = 1)
            self.size = out_size(self.size, 3, 1)
            self.pool1 = torch.nn.MaxPool2d(3)
            self.size = self.size // 3
            self.conv4 = torch.nn.Conv2d(1024, 2048, kernel_size = 3, stride = 1)
            self.size = out_size(self.size, 3, 1)
            self.pool2 = torch.nn.MaxPool2d(2)
            self.size = self.size // 2

            self.fc1 = nn.Linear(self.size*self.size*2048, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, set.num_catagories)

            self.bn1 = nn.BatchNorm1d(1024)
            self.bn2 = nn.BatchNorm1d(1024)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.pool1(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.pool2(x))
            x = x.view(x.size()[0], -1)
            x = self.bn1(F.relu(self.fc1(x)))
            x = self.bn2(F.relu(self.fc2(x)))
            x = F.log_softmax(self.fc3(x),-1)
            return x

        def dream(self, x):
            
            def deconv(x, weights):
                x_size = x.size() #(s, s, channels)
                weights_size =  weights.size() # (f, f, in_channels, out_channels)
                print("x size: ", x_size, "   w size: ",  weights_size)
                output = torch.zeros(weights_size[2], x_size[1], x_size[0]) # (channels, x, y)
                f = weights_size[0]
                pad = f // 2
                out_channels = weights_size[3]

                for i in range( x_size[1] - f):
                    for j in range(x_size[0] - f):
                        for k in range(out_channels):
                            output[:, i:(i+f), j:(j+f) ] = output[:, i:(i+f), j:(j+f) ] + weights[:,:,:,k] * x[i, j, k]

                output = output / out_channels

                return output


            filt1 = conv1.weight
            x = F.relu(self.conv1(x))

            


    print("moving model to device...")
    net = Net1().to(device)
    opt = optim.Adam(net.parameters(), lr = 0.0001)
    print("done")
    print()
    epochs = 64
    running_loss = 0.0

    for epoch in range(epochs):
        tic = time.time()
        for data in trainset:
            X, Y = data
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            #output = net(X.view(-1, 3,set.image_shape[0],set.image_shape[1]))
            output = net(X)
            loss = F.nll_loss(output, Y.squeeze_())
            loss.backward()
            running_loss += loss.item()
            opt.step()
        toc = time.time()
        print("Epoch: ", epoch+1,"  seconds: ", round(toc - tic, 3),"  ", end = '')
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testset:
                X, Y = data
                X, Y = X.to(device), Y.to(device)
                #output = net(X.view(-1, 3,set.image_shape[0],set.image_shape[1]))
                output = net(X)
                for idx, i in enumerate(output):
                    if torch.argmax(i) == Y[idx]:
                        correct += 1
                    total += 1
            print("acc: ", round(correct/total* 100, 3), "%  cost: ", running_loss * (batch_size / len(train_set)))
            #plot_filters(net.conv1.weight.cpu().numpy()[1])
            running_loss = 0.0

if __name__ == '__main__':
    run()