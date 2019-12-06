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
from set import TextureDataset

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dtd\\images\\')

set = TextureDataset(root)

train_set, dev_set = random_split(set, [len(set) - 500, 500])


trainset = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
testset = DataLoader(dev_set, batch_size=16, shuffle=True, num_workers=2)

print(len(dev_set))
print(len(train_set))
print(set.image_shape)
print(set.num_catagories)

def run():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
            self.fc1 = nn.Linear(set.image_shape[0]*set.image_shape[1]*3, 196)
            self.fc2 = nn.Linear(196, 128)
            self.fc3 = nn.Linear(128, set.num_catagories)

            self.bn1 = nn.BatchNorm1d(196)
            self.bn2 = nn.BatchNorm1d(128)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = x.view(x.size()[0], -1)
            x = self.bn1(F.relu(self.fc1(x)))
            x = self.bn2(F.relu(self.fc2(x)))
            x = F.log_softmax(self.fc3(x))
            return x

    net = Net().to(device)
    opt = optim.Adam(net.parameters(), lr = 0.0001)
    epochs = 5

    for epoch in range(epochs):
        tic = time.time()
        for data in trainset:
            X, Y = data
            X, Y = X.to(device), Y.to(device)
            output = net(X.view(-1, 3,set.image_shape[0],set.image_shape[1]))
            loss = F.nll_loss(output, Y.squeeze_())
            loss.backward()
            opt.step()
        toc = time.time()
        print("Epoch: ", epoch+1,"  seconds: ", round(toc - tic, 3),"  ", end = '')
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testset:
                X, Y = data
                X, Y = X.to(device), Y.to(device)
                output = net(X.view(-1, 3,set.image_shape[0],set.image_shape[1]))
                for idx, i in enumerate(output):
                    if torch.argmax(i) == Y[idx]:
                        correct += 1
                    total += 1
            print("acc: ", round(correct/total, 3))

if __name__ == '__main__':
    run()