import torch as torch
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
from Models import Model

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

def train():

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dtd/images/')
    set = TextureDataset(root)

    epochs = 16
    batch_size = 148
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
    
    dec_opt = optim.Adam(model.decoder.parameters(), weight_decay=1e-5)
    cls_opt = optim.Adam(model.classif.parameters(), weight_decay=1e-5)
    print("done")
    print()

    distance = nn.MSELoss()
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

            cls_loss = F.nll_loss(output, Y.squeeze_())
            cls_loss.backward()
            cls_opt.step()

            dec_opt.zero_grad()

            dec_loss = distance(decoding, X)
            dec_loss.backward()
            running_loss += dec_loss.item()
            dec_opt.step()


        toc = time.time()
        print("Epoch: ", epoch+1,"  seconds: ", round(toc - tic, 1),"  classifer_loss: ", round(cls_loss.item(), 2),"  decoder_loss: ", round(dec_loss.item(), 2)
    
    enc_name = "enc" + model_name
    dec_name = "dec" + model_name
    cls_name = "cls" + model_name

    torch.save(model.encoder.state_dict(), os.path.join(PATH, enc_name))
    torch.save(model.decoder.state_dict(), os.path.join(PATH, dec_name)) 
    torch.save(model.classif.state_dict(), os.path.join(PATH, cls_name))  

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