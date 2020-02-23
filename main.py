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

def output_to_image(out_img):

    out_img = out_img.squeeze_()
    out_img = out_img.cpu()
    out_img = out_img.detach().numpy()
    out_img = np.uint8(out_img*255)
    out_img = np.swapaxes(out_img, 0, 2)
    out_img = PIL.Image.fromarray(out_img)
    out_img = out_img.rotate(270, expand = True)
    out_img = out_img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    return out_img

def image_to_input(img):

    img = np.asarray(img, dtype= np.float32)
    img = np.transpose(img, (2,0,1))
    img = img / 255
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    img = img.to(device)

    return img

def train():

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dtd/images/')
    set = TextureDataset(root)

    epochs = 256
    batch_size = 128
    test_size = 500

    train_set, dev_set = random_split(set, [len(set) - test_size, test_size])
    trainset = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)
    testset = DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=12)

    print("dev set size:     ", len(dev_set))
    print("train set size:   ", len(train_set))
    print("image shape:      ", set.image_shape)
    print("image catagories: ", set.num_catagories)
    print("total images:     ", len(set))

    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/')
    model_name = str(input("model_name> "))

    print("moving model to device...")

    model = Model(set.image_shape[0])
    model.classif.to(device)
    model.decoder.to(device)
    model.encoder.to(device)
    
    dec_opt = optim.Adam(model.decoder.parameters())
    cls_opt = optim.Adam(model.classif.parameters(), weight_decay=1e-5)
    print("done")
    print()

    distance = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    loss_dict = {"cls": [], "dec": []}
    for epoch in range(epochs):
        tic = time.time()
        for data in trainset:
            X, Y = data
            X, Y = X.to(device), Y.to(device)
            
            cls_opt.zero_grad()
            encoding = model.encoder(X)
            decoding = model.decoder(encoding)
            output = model.classif(encoding)

            cls_loss = criterion(output, Y.squeeze_())
            cls_loss.backward(retain_graph = True)
            cls_opt.step()

            dec_opt.zero_grad()

            dec_loss = distance(decoding, X)
            dec_loss.backward()
            dec_opt.step()
            
            #loss_dict["cls"] = loss_dict["cls"].append(cls_loss.item())
            #loss_dict["dec"] = loss_dict["dec"].append(dec_loss.item())


        toc = time.time()
        print("Epoch: ", epoch+1,"  seconds: ", round(toc - tic, 1),"  classifer_loss: ", round(cls_loss.item(), 4),"  decoder_loss: ", round(dec_loss.item(), 4))
    
    enc_name = "enc_" + model_name
    dec_name = "dec_" + model_name
    cls_name = "cls_" + model_name

    torch.save(model.encoder.state_dict(), os.path.join(PATH, enc_name))
    torch.save(model.decoder.state_dict(), os.path.join(PATH, dec_name)) 
    torch.save(model.classif.state_dict(), os.path.join(PATH, cls_name))
    print("saved !")
    print()

def test():

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dtd/images/')
    set = TextureDataset(root)

    testset = DataLoader(set, batch_size=1, shuffle=True, num_workers=8)

    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/')
    model_name = str(input("model_name> "))

    enc_name = "enc_" + model_name
    dec_name = "dec_" + model_name
    cls_name = "cls_" + model_name

    print("moving model to device...")

    model = Model(256)

    model.decoder.load_state_dict(torch.load(os.path.join(PATH, dec_name)))
    model.decoder.to(device)
    model.decoder.eval()

    model.encoder.load_state_dict(torch.load(os.path.join(PATH, enc_name)))
    model.encoder.to(device)
    model.encoder.eval()

    model.classif.load_state_dict(torch.load(os.path.join(PATH, cls_name)))
    model.classif.to(device)
    model.classif.eval()

    tallies = set.get_dict()
    print("image shape:      ", set.image_shape)
    print("image catagories: ", set.num_catagories)
    print("total images:     ", len(set))

    with torch.no_grad():
        for data in testset:
            X, Y = data
            X, Y = X.to(device), Y.to(device)


            encoding = model.encoder(X)
            output = model.classif(encoding)

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

    enc_name = "enc_" + model_name
    dec_name = "dec_" + model_name
    cls_name = "cls_" + model_name

    print("moving model to device...")
    model = Model(256)
    model.decoder.load_state_dict(torch.load(os.path.join(PATH, dec_name)))
    model.decoder.to(device)
    model.decoder.eval()
    model.encoder.load_state_dict(torch.load(os.path.join(PATH, enc_name)))
    model.encoder.to(device)
    model.encoder.eval()
    print("done!")
    print("dream functions: ")

    test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_images/')
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_outputs/')
    test_images = os.listdir(test_folder)
    print("pulling images from : ", test_folder)
    print("dumping to          : ", output_folder)
    print("counting sheep...")

    for image in test_images:
        print("     ", image)
        img = PIL.Image.open(os.path.join(test_folder, image))
        model_input = image_to_input(img)
        model_output = model.noise_injection(model_input)
        dream_image = output_to_image(model_output)
        
        dream_image.save(os.path.join(output_folder, "dream_" + image))

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