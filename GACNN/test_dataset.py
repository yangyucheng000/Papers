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


class MyDataset2(Dataset):
    def __init__(self, resize=None, datakind=None):

        trans = []
        self.image = []
        self.label = []
        if resize:
            trans.append(torchvision.transforms.Resize(size=resize))
        trans.append(torchvision.transforms.ToTensor())
        self.transform = torchvision.transforms.Compose(trans)
        self.loaddata(datakind)

    def __len__(self):
        return len(self.image)

    def loaddata(self, datakind):
        path = r"/tmp/code/gacnn/datasets1/test_0"
        path0 = path + "/" + datakind
        #print(path0)
        filelist0 = glob.glob(os.path.join(path0, "*.png"))
        for each in filelist0:

            img = Image.open(each)
            if self.transform:
                img = self.transform(img)

            self.image.append(img)
            self.label.append(1)


    def __getitem__(self, index):

        sample = {'data': self.image[index], 'label': self.label[index]}
        return sample


if __name__ == '__main__':
    data = MyDataset2(244, 0)
    print(1)
