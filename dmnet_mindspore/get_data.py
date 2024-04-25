import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore as ms
from mindspore import Tensor
from PIL import ImageEnhance
import random
import os
from PIL import Image
import numpy as np
import utils
import time

def dogs_folders(split):
    opt = utils.get_global_config()
    data_dir = opt['data_dir']
    if split=='train':
        folder = os.path.join(data_dir, 'train')
    elif split=='test':
        folder = os.path.join(data_dir, 'test')
    elif split=='val':
        folder = os.path.join(data_dir, 'val')
    else: print('error in task_generator')

    class_folders = [os.path.join(folder, label) for label in os.listdir(folder)]
    labels = np.array(range(len(class_folders)), dtype=np.int32)

    labels_dict = dict(zip(class_folders, labels))

    image_roots = []
    image_segment_cnt = []
    for c in class_folders:
        image_roots.extend([os.path.join(c, x) for x in os.listdir(c)])
        image_segment_cnt.append(len(image_roots))

    return folder, labels_dict, image_roots, image_segment_cnt

class Dataset:
    def __init__(self, image_roots, labels_dict, image_cnt, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.len = image_cnt
        self.images = []
        self.labels = []
        for image_root in image_roots:
            image = Image.open(image_root) #HWC
            image = image.convert('RGB')
            if self.transform is not None:
                image = self.transform(image)[0] #tuple after transform 1*3*84*84
            
            label = labels_dict[os.path.split(image_root)[0]]
            if self.target_transform is not None:
                label = self.target_transform(label)
            self.images.append(image)
            self.labels.append(label)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class Sampler:
    def __init__(self, image_segment_cnt, way, shot, query):
        self.way = way
        self.shot = shot
        self.query = query
        self.i_s_c = [0]+image_segment_cnt
        self.total_way = len(image_segment_cnt)

    def __iter__(self):
        r_way = random.sample(range(self.total_way), k = self.way)
        r_images = []
        for way in r_way:
            r_images.append(random.sample(range(self.i_s_c[way], self.i_s_c[way+1]), k = self.shot+self.query))
        for sublist in r_images:
            random.shuffle(sublist)

        idx = []
        idxb = []
        for i in range(self.way): idx.extend(r_images[i][:self.shot])
        for i in range(self.way): idxb.extend(r_images[i][self.shot:])
        random.shuffle(idxb)
        idx.extend(idxb)

        return iter(idx)

    def __len__(self):
        return self.way*(self.shot+self.query)

class Data_Prepare():
    def __init__(self, split='train'):
        self.opt = utils.get_global_config()
        self.split = split
        self.dataset_name = self.opt['dataset_name']

        self.jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)

        assert self.dataset_name=='dogs'
        self.floder, self.labels_dict, self.image_roots, self.image_segment_cnt = dogs_folders(split)
        self.size = 84
        if split=='test' or split=='val':
            # self.transform = transforms.Compose([transforms.ToTensor()])
            # to be revised   

            self.transform = transforms.Compose([
                                            vision.Resize([int(self.size*1.15), int(self.size*1.15)]),
                                            vision.CenterCrop(self.size),
                                            vision.ToTensor(), #HWC->CHW
                                            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)])

        else:
            self.transform = transforms.Compose([
                                            vision.RandomResizedCrop(self.size),
                                            # ImageJitter(self.jitter_param),
                                            vision.RandomHorizontalFlip(),
                                            vision.ToTensor(),
                                            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)])

        self.split = split

        self.target_transform = None

    def get_loader(self):
        s_sampler = Sampler(self.image_segment_cnt, self.opt['way'], self.opt['shot'], self.opt['query'])
        print('Generating dataset', self.split, end='    ')
        t = time.time()
        s_dataset = Dataset(self.image_roots, self.labels_dict, self.image_segment_cnt[-1], self.transform, self.target_transform)
        print(f'time cost is {time.time()-t:0.6f} s')
        dataset = ds.GeneratorDataset(s_dataset, column_names=["data", "label"], sampler=s_sampler)

        dataset = dataset.batch(self.opt['way']*(self.opt['shot']+self.opt['query']))

        return dataset

class ImageJitter(object):
    def __init__(self, transformdict):
        transformtypedict = dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = ops.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')
        return out