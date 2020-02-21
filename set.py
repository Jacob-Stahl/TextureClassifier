import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from scipy import misc
import numpy as np
import imageio as io
from PIL import Image

class TextureDataset(Dataset):
    def __init__(self, root, transform = None):
        self.transform = transform
        self.image_shape = (256, 256)
        self.image_paths = []
        self.image_labels = []
        self.num_catagories = 0
        self.dict_ = {}
        label = 0
        for texture in os.listdir(root):

            self.dict_[texture] = 0
            texture_folder = os.path.join(root, texture)

            for image in os.listdir(texture_folder):
                image_path = os.path.join(texture_folder, image)
                self.image_paths.append(image_path)
                self.image_labels.append(label)

            label += 1
        self.num_catagories = label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path)
        image = image.resize(self.image_shape)

        image = np.asarray(image, dtype= np.float32)
        image = np.transpose(image, (2,0,1))
        image = image / 255

        image = torch.from_numpy(image).float()

        sample = (image, self.image_labels[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample

    def show_item(self, idx):
        image_path = self.image_paths[idx]

        label = np.zeros((self.num_catagories, 1))
        label[self.image_labels[idx]] = 1
        image = Image.open(image_path)
        image = image.resize(self.image_shape)

        image.show()

    def get_dict(self):

        return self.dict_


if __name__ == '__main__':

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dtd/images/')

    trans = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees = 90),
    transforms.ToTensor(),
])
    set = TextureDataset(root, trans)

    dataloader = DataLoader(set, batch_size=5, shuffle=True, num_workers=2)    
    