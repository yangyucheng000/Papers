# -*- coding: utf-8 -*-
# 瀵煎叆鐩稿叧妯″潡
import torchvision
from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import glob
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, resize=None,datakind=None):
        trans = []
        self.image=[]
        self.label=[]
        if resize:
            trans.append(torchvision.transforms.Resize(size=resize))
            # trans.append(torchvision.transforms.CenterCrop(size=resize))
        trans.append(torchvision.transforms.ToTensor())
        self.transform = torchvision.transforms.Compose(trans)
        self.loaddata(datakind)

    def __len__(self):
        return len(self.image)

    def loaddata(self,datakind):
        path=r"/tmp/code/gacnn/datasets1/"+datakind
        path0=path+"/0"
        filelist0=glob.glob(os.path.join(path0, "*.png"))
        for each in filelist0:
            img = Image.open(each)
            if self.transform:
                img = self.transform(img)
                # print(img.size())
            self.image.append(img)
            self.label.append(0)
        path1=path+"/1"
        filelist1=glob.glob(os.path.join(path1, "*.png"))
        for each in filelist1:
            img = Image.open(each)
            if self.transform:
                img = self.transform(img)

            self.image.append(img)
            self.label.append(1)

    def __getitem__(self, index):

        sample = {'data': self.image[index], 'label': self.label[index]}
        return sample

if __name__ == '__main__':
    data=MyDataset(224,0)
    print(1)
