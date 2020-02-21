import torch
import torch.optim as optim
import torchvision 
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import pprint as pp
import numpy as np
import PIL

from set import TextureDataset

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")

def pprint_dict(dict_):
    for key in dict_:
        print(dict_[key]," : ", key)

class Encoder(nn.Module):

    def __init__(self, encoding_channels, dim):

        super().__init__()
        self.encoding_channels = encoding_channels
        self.dim = dim # keeps track of tensor WxH as it is passed down the network

        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size= 5, stride= 1, padding= 2)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.pool1 = torch.nn.MaxPool2d(kernel_size= 2)
        self.dim = dim // 2
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size= 5, stride= 1, padding= 2)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.pool2 = torch.nn.MaxPool2d(kernel_size= 2)
        self.dim = dim // 2
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

        self.conv1 = torch.nn.ConvTranspose2d(self.encoding_channels, 16, kernel_size= 5, stride= 1, padding= 2)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.pool1 = torch.nn.MaxPool2d(kernel_size= 2)
        self.conv2 = torch.nn.ConvTranspose2d(16, 8, kernel_size= 5, stride= 1, padding= 2)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.pool2 = torch.nn.MaxPool2d(kernel_size= 2)
        self.conv3 = torch.nn.ConvTranspose2d(8, 3, kernel_size= 5, stride= 1, padding= 2)
        self.conv4 = torch.nn.Conv2d(3, 3, kernel_size= 1, stride= 1, padding= 0)

    def forward(self, x):

        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x))
        x = self.pool2(x)
        x = F.sigmoid(self.conv3(x))
        x = self.conv4(x)

        return x

class Classifier(nn.Module):
    def __init__(self, encoding_channels, dim):

        super().__init__()
        self.encoding_channels = encoding_channels
        self.dim = dim

        self.conv1 = torch.nn.Conv2d(self.encoding_channels, 16, kernel_size = 3, stride = 1, padding= 1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.pool1 = torch.nn.MaxPool2d(kernel_size = 2)
        self.dim = self.dim // 2
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding= 1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.pool2 = torch.nn.MaxPool2d(kernel_size = 2)
        self.dim = self.dim // 2
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding= 1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.pool3 = torch.nn.MaxPool2d(kernel_size = 2)
        self.dim = self.dim // 2
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding= 1)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.pool4 = torch.nn.MaxPool2d(kernel_size = 2)
        self.dim = self.dim // 2
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

        enc1 = Encoder(img1)
        enc2 = Encoder(img2)

        squash = (enc1, enc2) / 2

        img_out =  Decoder(squash)

        return img_out


def train():

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dtd/images/')
    set = TextureDataset(root)

    epochs = 128
    batch_size = 64
    learning_rate = 0.00005
    test_size = 500

    train_set, dev_set = random_split(set, [len(set) - test_size, test_size])
    trainset = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    testset = DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=4)

    print("dev set size:     ", len(dev_set))
    print("train set size:   ", len(train_set))
    print("image shape:      ", set.image_shape)
    print("image catagories: ", set.num_catagories)
    print("total images:     ", len(set))

    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/')
    model_name = str(input("model_name> "))
    PATH =  os.path.join(PATH, model_name)

    print("moving model to device...")

    model = Model(set.image_shape[0])
    model.classif.to(device)
    model.decoder.to(device)
    model.encoder.to(device)
    
    dec_opt = optim.Adam(model.decoder.parameters(), lr=learning_rate)
    cls_opt = optim.Adam(model.classif.parameters(), lr=learning_rate)
    print("done")
    print()

    running_loss = 0.0
    for epoch in range(epochs):
        tic = time.time()
        for data in trainset:
            X, Y = data
            X, Y = X.to(device), Y.to(device)
            
            cls_opt.zero_grad()
            encoding = model.encoder(X)
            decoding = model.decoder(X)
            output = model.classif(encoding)

            print(output.size)

            cls_loss = F.nll_loss(output, Y.squeeze_())
            cls_loss.backward()
            cls_opt.step()

            dec_opt.zero_grad()

            loss = F.nll_loss(decoding, X)
            loss.backward()
            running_loss += loss.item()
            dec_opt.step()


        toc = time.time()
        print("Epoch: ", epoch+1,"  seconds: ", round(toc - tic, 1),"  ", end = '')
    
    enc_name = "enc" + model_name
    dec_name = "dec" + model_name
    cls_name = "cls" + model_name

    torch.save(model.encoder.state_dict(), os.path.join(PATH, enc_name))
    torch.save(model.decoder.state_dict(), os.path.join(PATH, dec_name)) 
    torch.save(model.classif.state_dict(), os.path.join(PATH, cls_name))  


def old_train():

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dtd/images/')
    set = TextureDataset(root)

    epochs = 128
    batch_size = 128
    learning_rate = 0.00005
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
    print("total images:     ", len(set))

    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/')
    model_name = str(input("model_name> "))
    PATH =  os.path.join(PATH, model_name)

    print("moving model to device...")
    net = Net1().to(device)
    opt = optim.Adam(net.parameters(), lr = learning_rate, weight_decay= 0.01)
    print("done")
    print()
    running_loss = 0.0

    record = 0
    for epoch in range(epochs):
        tic = time.time()
        for data in trainset:
            X, Y = data
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            output = net(X)
            loss = F.nll_loss(output, Y.squeeze_())
            loss.backward()
            running_loss += loss.item()
            opt.step()
        toc = time.time()
        print("Epoch: ", epoch+1,"  seconds: ", round(toc - tic, 1),"  ", end = '')
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testset:
                X, Y = data
                X, Y = X.to(device), Y.to(device)
                output = net(X)
                for idx, i in enumerate(output):
                    if torch.argmax(i) == Y[idx]:
                        correct += 1
                    total += 1

            acc = round(correct/total* 100, 3)
            print("acc: ", acc, "%  cost: ", running_loss * (batch_size / len(train_set)), end = "")
            if acc > record:
                record = acc
                print(" saved! ", end = "")
                torch.save(net.state_dict(), PATH +"_"+str(acc)+"_"+str(running_loss))
            if running_loss < 0.1:
                record = acc
                print(" saved! ", end = "")
                torch.save(net.state_dict(), PATH +"_"+str(acc)+"_"+str(running_loss))
                return
            print()
                

            #plot_filters(net.conv1.weight.cpu().numpy()[1])
            running_loss = 0.0

    torch.save(net.state_dict(), PATH)

def test():

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dtd/images/')
    set = TextureDataset(root)

    testset = DataLoader(set, batch_size=56, shuffle=True, num_workers=4)

    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/')
    model_name = str(input("model_name> "))
    PATH =  os.path.join(PATH, model_name)

    net = Net1().to(device)
    net.load_state_dict(torch.load(PATH))
    net.eval()

    tallies = set.get_dict()
    print("image shape:      ", set.image_shape)
    print("image catagories: ", set.num_catagories)
    print("total images:     ", len(set))

    with torch.no_grad():
        for data in testset:
            X, Y = data
            X, Y = X.to(device), Y.to(device)
            output = net(X)
            for idx, i in enumerate(output):
                from_hot = torch.argmax(i)
                if from_hot == Y[idx]:
                    tallies[list(tallies.keys())[from_hot]] += 1

    pprint_dict(tallies)
def dream_test():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dtd/images/')
    set = TextureDataset(root)

    testset = DataLoader(set, batch_size=56, shuffle=True, num_workers=4)

    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/')
    model_name = str(input("model_name> "))
    PATH =  os.path.join(PATH, model_name)

    net = Net1()
    net.load_state_dict(torch.load(PATH))
    net.to(device)
    net.eval()

    for i in [0, 10, 20, 69, 4000, 600]:

        input_img = set[i][0].unsqueeze_(0)
        print(input_img.size())
        out_img = net(input_img.to(device), dream = True)
        print(out_img.size())
        out_img = out_img.squeeze_()
        print(out_img.size())
        out_img = out_img.cpu()
        print(out_img.size())
        out_img = out_img.detach().numpy()
        out_img = np.uint8(out_img*255)
        out_img = np.swapaxes(out_img, 0, 2)
        out_img = PIL.Image.fromarray(out_img)

        out_img.show()

if __name__ == '__main__':

    option  = -1
    while option != 0:

        print("0 - quit")
        print("1 - train")
        print("2 - test")
        print("3 - dream")

        option  = int(input("> "))
        if option == 0:
            break
        if option == 1:
            train()
        if option == 2:
            test()
        if option == 3:
            dream_test()