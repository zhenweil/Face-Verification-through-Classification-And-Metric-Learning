from torch.utils.data import Dataset
from torchvision import transforms, datasets
import os
from PIL import Image
import numpy as np

class MyVerificationDataset(Dataset):
    def __init__(self, FilePath, transform):
        files = open(FilePath).read().splitlines()
        self.is_test = False
        self.transform = transform
        self.file_list = []
        for i in range(len(files)):
            verification_pair = files[i].split(" ")
            self.file_list.append(verification_pair)
        if(len(self.file_list[0]) == 2):
            self.is_test = True
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        image1 = Image.open("../" + self.file_list[index][0])
        image2 = Image.open("../" + self.file_list[index][1])
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        if(self.is_test == False):
            label = self.file_list[index][2]
            return image1, image2, int(label)
        else:
            return image1, image2