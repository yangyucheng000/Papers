# -*- coding: utf-8 -*-
import torch
import os
import torch.nn as nn
import collections
from resnet import resnet18
from compare import get_testimg, getindex
from distance import getdistance
import random
import json

from treelib import Tree, Node


class NewModel(nn.Module):
    def __init__(self,task_num=None,order=None):
        super(NewModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.allmodel = {}
        self.trees={}
        self.totalkeys = {}
        self.order=order[0:task_num]
        # self.order = random.sample(range(0, 10), 10)  # 随机生成顺序
        print(self.order)
        self.initmodels()
        # collections.OrderedDict()

    def initmodels(self):  # 修改合并网络及网络存放地址
        path = './model_fine_100/'
        dir = os.listdir(path)

        k = 0
        for o in self.order:
            for i in dir:
                j = i[:-4]
                #print(j)
                if j[6:] == str(o):
                    model = resnet18(num_classes=2)
                    model.cuda()
                    model.eval()
                    trained_weight = torch.load(path + i)
                    model.load_state_dict(trained_weight)
                    self.allmodel[k] = model

                    self.initmodel(modelpath=path + 'model_' + str(o) + '.pth', modelkind=k)
                    k=k+1
                    break
        #print(self.trees[k-1])


    def initmodel(self, modelpath, modelkind):
        model = resnet18(num_classes=2)
        trained_weight = torch.load(modelpath)
        model.load_state_dict(trained_weight)
        #model = myvgg.get_trained_vgg(path=modelpath)
        self.cuda()
        model.cuda()
        model.eval()
        self.eval()



    def forward(self, x):
        res = {}
        for k in self.allmodel.keys():
            res[k] = self.allmodel[k](x)  # all path results
            res[k]= res[k].softmax(1)
        ans = self.parseres(res)
        return ans

    def parseres(self, res):
        key = []
        for i in res.keys():
            key.append(i)

        result = [-1 for i in range(res[key[0]].shape[0])]

        for i in range(res[key[0]].shape[0]):
            max = -100
            max1=-100
            result[i] = 0
            for k in res.keys():
                if res[k][i][1] > 0 and res[k][i][1]  > max: # 对于正数的结果选差值最大的,结果0.6455
                    max=res[k][i][1]
                    result[i]=self.order[k]

        return result


if __name__ == '__main__':
    model = NewModel()
    img = get_testimg(path=r"./datasets/test/0.png",
                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    res = model(img)


