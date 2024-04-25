import os
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.dataset as de
from mindspore.dataset.vision import Inter as Inter
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import pandas as pd
import cv2
from PIL import Image
from mindspore import Tensor

class ERADataset:
    def __init__(self, root='/home/tbd/tdwc/dataset/msp_data/MSP/yg/', filename="train4.csv"):
        image_dir = os.path.join(root, "Test")
        filename = os.path.join(root, filename)
        dataset = pd.read_csv(filename)
        self.images = []
        self.labels = []
        for iname, itype in np.array(dataset).tolist():
            if iname.endswith(".png"):
                try:
                    image_path = os.path.join(image_dir, iname)
                    img = cv2.imread(image_path).astype(np.float32)
                    # img = cv2.resize(cv2.imread(image_path).astype(np.float32), (256, 256))
                    self.images.append(img)
                    self.labels.append(itype)
                except:
                    continue
        self.transform = self.train_transform

    def train_transform(self, image, label):
        img = cv2.resize(image, (256, 256))
        img = img.transpose((2, 0, 1))  #  --> C H W
        img = np.array(img, dtype=np.float32) / np.float32(255.0)
        return img, label

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        img, label = self.transform(img, label)
        return img, label

    def __len__(self):
        return len(self.images)
# ==================================

def create_dataset(phase="train", batch_size=100, device_num=1, rank=0):
    if phase == "train":
        filename="train4.csv"
        do_shuffle = True
    else:
        filename = "test4.csv"
        do_shuffle = False
    dset = ERADataset(filename=filename)
    onehot_op = transforms.OneHot(num_classes=25)
    ds = de.GeneratorDataset(dset, column_names=["data", "label"], num_parallel_workers=8, shuffle=do_shuffle)
    ds = ds.map(input_columns="data", operations=transforms.TypeCast(ms.float32))
    ds = ds.map(operations=onehot_op, input_columns=["label"])
    ds = ds.map(input_columns="label", operations=transforms.TypeCast(ms.uint8))

    ds = ds.batch(batch_size)
    return ds
