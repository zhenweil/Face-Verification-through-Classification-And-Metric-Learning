from torch.utils.data import Dataset
from torchvision import transforms, datasets
import os
from PIL import Image
import numpy as np
import random

class train_dataset(Dataset):
    def __init__(self, root, images_per_person, transform):
        self.images_per_person = images_per_person
        self.transform = transform
        filenames = [os.path.join(root, f.name) for f in os.scandir(root)]
        self.labels = [f.name for f in os.scandir(root)]
        imgs = []
        for f in filenames:
            imgs.append([os.path.join(f, sub_f.name) for sub_f in os.scandir(f)])
        self.train_pairs = []
        num_class = len(imgs)
        for i in range(num_class):
            for j in range(self.images_per_person):
                anchor = imgs[i][j]
                positive_idx = random.randint(j+1,j+num_img-1)%num_img
                positive = imgs[i][positive_idx]
                negative_class = random.randint(i+1, i+num_class-1)%num_class
                num_image_negative = len(imgs[negative_class])
                negative_idx = random.randint(0, num_image_negative - 1)
                negative = imgs[negative_class][negative_idx]
                self.train_pairs.append([anchor, positive, negative])
                
    def __len__(self):
        return len(self.train_pairs)
    
    def __getitem__(self, idx):
        anchor_path = self.train_pairs[idx][0]
        positive_path = self.train_pairs[idx][1]
        negative_path = self.train_pairs[idx][2]
        anchor = Image.open(anchor_path)
        positive = Image.open(positive_path)
        negative = Image.open(negative_path)
        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative = self.transform(negative)
        return anchor, positive, negative

class dev_dataset(Dataset):
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