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
        if task_num!=None:
            self.order=order[0:task_num]
        else:
            self.order = random.sample(range(0, 100), 100)  # 随机生成顺序
        print(self.order)
        self.initmodels()
        # collections.OrderedDict()

    def initmodels(self):  # 修改合并网络及网络存放地址
        path = '/tmp/code/gacnn/model/'
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
                    tree = Tree()
                    tree.create_node(j[6:] + '-1', j[6:] + '-1')  # 根节点
                    # print(tree)
                    tree.create_node(j[6:] + '-2', j[6:] + '-2', parent=j[6:] + '-1')
                    tree.create_node(j[6:] + '-3', j[6:] + '-3', parent=j[6:] + '-2')
                    tree.create_node(j[6:] + '-4', j[6:] + '-4', parent=j[6:] + '-3')
                    tree.create_node(j[6:] + '-5', j[6:] + '-5', parent=j[6:] + '-4')
                    tree.create_node(j[6:] + '-6', j[6:] + '-6', parent=j[6:] + '-5')
                    tree.create_node(j[6:] + '-7', j[6:] + '-7', parent=j[6:] + '-6')
                    tree.create_node(j[6:] + '-8', j[6:] + '-8', parent=j[6:] + '-7')
                    self.trees[k] = tree
                    self.initmodel(modelpath=path + 'model_' + str(o) + '.pth', modelkind=k)
                    k=k+1
                    break
        print(self.trees[k-1])


    def initmodel(self, modelpath, modelkind):
        model = resnet18(num_classes=2)
        trained_weight = torch.load(modelpath)
        model.load_state_dict(trained_weight)
        #model = myvgg.get_trained_vgg(path=modelpath)
        self.cuda()
        model.cuda()
        model.eval()
        self.eval()
        resdict = {}
        if modelkind>0:
          # 新网络与每一个路径计算共享层索引值
            img = get_testimg(path=r"/tmp/code/gacnn/datasets1/test_0/0/apple_s_000022.png", device=self.device)
            for k in self.allmodel.keys():
                #print('k:',k)
                if k<modelkind:
                    t = self.allmodel[k]#其他9种模型
                    out1 = t.forward_per_layer(img)
                    out2 = model.forward_per_layer(img)
                    dis = getdistance(out1[1], out2[1])
                    index = getindex(dis, thre=0.86)  # 可复用的模块
                    resdict[k] = index  # 与现有每一条路径比较获得全部插入层数

            model_inserted = max(resdict, key=resdict.get)  # 返回新任务网络最终插入的路径号key，选择最相似路径即最高层所在路径
            idx = resdict[model_inserted]  # 插入该路径的层数

            mergepoint=str(self.order[model_inserted]) + '-'+str(idx)
            #print(mergepoint)
            if idx==0:
                idx=idx+1
                mergepoint = str(self.order[0]) + '-' + str(idx)
            idx1=idx
            if not self.trees[model_inserted].contains(mergepoint):
                mergepoint = str(self.order[0]) + '-' + str(idx)
                idx1=mergepoint
            #print("mergepoint:",mergepoint)

            print("put {} model insert in the {}th layel of {} model".format(modelkind, idx, model_inserted))
            #print(self.trees[modelkind])
            tree1=self.trees[modelkind]
            #print(tree1)
            tree2=self.trees[model_inserted]

            tree3=tree1.subtree(str(self.order[modelkind])+ '-'+str(idx))
            tree2.merge(mergepoint, tree3)

            for k in self.allmodel.keys():

                if k <= modelkind:
                    self.trees[k] = tree2
                #print(self.trees[k])
            newmodel = self.mergetwomodel(self.allmodel[model_inserted], model, idx )

            self.allmodel[modelkind] = newmodel


    def mergetwomodel(self, oldmodel, newmodel, index):  # 合并新任务网络与路径
        oldmodel = oldmodel.state_dict()
        modeldic = newmodel.state_dict()

        block1 = [
                  'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean',
                  'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight',
                  'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var',
                  'layer1.0.bn2.num_batches_tracked']
        block2 = ['layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean',
                  'layer1.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight',
                  'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var',
                  'layer1.1.bn2.num_batches_tracked']
        block3 = ['layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean',
                  'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight',
                  'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var',
                  'layer2.0.bn2.num_batches_tracked', 'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.weight',
                  'layer2.0.downsample.1.bias', 'layer2.0.downsample.1.running_mean',
                  'layer2.0.downsample.1.running_var', 'layer2.0.downsample.1.num_batches_tracked']
        block4 = ['layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean',
                  'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.conv2.weight',
                  'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var',
                  'layer2.1.bn2.num_batches_tracked']
        block5 = ['layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean',
                  'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked', 'layer3.0.conv2.weight',
                  'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var',
                  'layer3.0.bn2.num_batches_tracked', 'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.weight',
                  'layer3.0.downsample.1.bias', 'layer3.0.downsample.1.running_mean',
                  'layer3.0.downsample.1.running_var', 'layer3.0.downsample.1.num_batches_tracked']
        block6 = ['layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean',
                  'layer3.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight',
                  'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var',
                  'layer3.1.bn2.num_batches_tracked']
        block7 = ['layer4.0.conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean',
                  'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.conv2.weight',
                  'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var',
                  'layer4.0.bn2.num_batches_tracked', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight',
                  'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean',
                  'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.num_batches_tracked']
        block8 = ['layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean',
                  'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv2.weight',
                  'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var',
                  'layer4.1.bn2.num_batches_tracked',
                  ]



        id = [block1,block2,block3,block4,block5,block6,block7,block8]
        if index <= 8:
            for each in id:
                if id.index(each) < index:  # 替换当前网络路径的参数给新任务网络
                    for i in each:
                        modeldic[i] = oldmodel[i]
        newmodel.load_state_dict(modeldic)
        return newmodel



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
                if res[k][i][1]  > max : # 对于正数的结果选差值最大的,结果0.6455
                    max=res[k][i][1]
                    result[i]=self.order[k]

        return result


if __name__ == '__main__':
    model = NewModel()
    img = get_testimg(path=r"/tmp/code/gacnn/datasets/test/0.png",
                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    res = model(img)


