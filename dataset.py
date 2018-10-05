import glob
import pandas as pd
import torch
import pdb
import os
import numpy as np
import PIL
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from config import *

def data_loader(args, mode='TRAIN'):
    if mode == 'TRAIN':
        shuffle = True
    elif mode == 'TEST':
        shuffle = False
    else:
        raise ValueError('data_loader flag ERROR')

    dataset = Dataset(args, mode)
    dataloader = DataLoader(Dataset(args, mode),
                            batch_size=args.batch_size,
                            num_workers=os.cpu_count(),
                            shuffle=shuffle,
                            drop_last=True)
    return dataloader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, mode):
        self.mode = mode
        self.img_root = args.img_root
        self.img_path = []
        self.label_path = args.label_path
        self.shape_list = []
        self.color1_list = []
        self.color2_list = []
        self.transform = transforms.Compose([
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(degrees=(-20,20),translate=(0.1,0.1),
                                            scale=(0.9,1.1), shear=(-0.2,0.2)),
                    transforms.ToTensor()])
        img_list = os.listdir(self.img_root)
        img_list = sorted(img_list, key=lambda x: int(os.path.splitext(x)[0]))
        for img_name in img_list:
            self.img_path += glob.glob(os.path.join(self.img_root, img_name))

        xls = pd.read_excel(self.label_path)
        for i in range(len(xls['shape'])):
            self.shape_list.append(shapeConvert(xls['shape'][i]))
            self.color1_list.append(colorConvert(xls['color_front'][i].split(',')[0]))
            if isinstance(xls['color_back'][i], float):
                self.color2_list.append(colorConvert(xls['color_front'][i].split(',')[0]))
            else:
                self.color2_list.append(colorConvert(xls['color_back'][i].split(',')[0]))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        if self.mode == 'TRAIN':
            img = self.transform(img)
            shape = self.shape_list[idx]
            color1 = self.color1_list[idx]
            color2 = self.color2_list[idx]
            return img, shape, color1, color2, self.img_path[idx]
        else:
            return img, self.img_path[idx]

