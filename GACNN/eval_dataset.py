# 导入相关模块
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

class MyDataset1(Dataset):
    def __init__(self, resize=None,order=None,task_num=None):
        self.task_num=task_num
        trans = []
        self.image=[]
        self.label=[]
        if resize:
            trans.append(torchvision.transforms.Resize(size=resize))
            # trans.append(torchvision.transforms.CenterCrop(size=resize))  �裁
        trans.append(torchvision.transforms.ToTensor())
        self.transform = torchvision.transforms.Compose(trans)

        self.loaddata(order)

    def __len__(self):
        return len(self.image)

    def loaddata(self,order):
        if order != None:
            for i in range(0,self.task_num):
                path=r"/tmp/code/gacnn/datasets1/test_0/"
                path0 = path + str(order[i])
                filelist0 = glob.glob(os.path.join(path0, "*.png"))
                for each in filelist0:
                    img = Image.open(each)
                    if self.transform:
                        img = self.transform(img)
                    self.image.append(img)
                    self.label.append(order[i])
        else:
            path = r"/tmp/code/gacnn/datasets1/test_0"
            dir = os.listdir(path)
            #k=0
            for i in sorted(dir):
                path0 = path + "/"+i
                filelist0 = glob.glob(os.path.join(path0, "*.png"))
                for each in filelist0:
                    img = Image.open(each)
                    if self.transform:
                        img = self.transform(img)
                    self.image.append(img)
                    self.label.append(int(i))
                #k=k+1






    def __getitem__(self, index):

        sample = {'data': self.image[index], 'label': self.label[index]}
        return sample

if __name__ == '__main__':
    data=MyDataset1(224,0)
    print(1)
