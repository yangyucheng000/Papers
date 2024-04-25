import torch
import os
import torch.nn as nn
import collections
from resnet import resnet18

import random
import json
from torch.utils import data
from torch.autograd import Variable

import eval_dataset

class NewModel(nn.Module):
    def __init__(self,newpath):
        super(NewModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.allmodel = {}
        self.trees = {}
        self.totalkeys = {}
        self.tree_json = {}
        #self.paths=paths
        self.res={}

        self.initmodels(newpath)

        # collections.OrderedDict()

    def initmodels(self,newpath):  # 修改合并网络及网络存放地址
        path = newpath
        dir = os.listdir(path)
        #print(len(dir))
        k = 0
        for i in sorted(dir):
            model = resnet18(num_classes=2)
            model.cuda()
            model.eval()
            trained_weight = torch.load(path + i)
            model.load_state_dict(trained_weight)
            self.allmodel[k] = model

            k = k + 1

        k = 0

        for i in sorted(dir):
            self.initmodel(modelpath=path + i, modelkind=k)
            k = k + 1





    def initmodel(self, modelpath, modelkind):
        model = resnet18(num_classes=2)
        trained_weight = torch.load(modelpath)
        model.load_state_dict(trained_weight)
        self.cuda()
        model.cuda()
        model.eval()
        self.eval()

    def cal_ans(self,k,x):
        self.res[k]=self.allmodel[k](x)


    def forward(self, x):
        for k in self.allmodel.keys():
            self.res[k] = self.allmodel[k](x)
        ans = self.parseres()
        return ans

    def parseres(self):
        key = []
        for i in self.res.keys():
            key.append(i)

        result = [-1 for i in range(self.res[key[0]].shape[0])]

        for i in range(self.res[key[0]].shape[0]):  #
            # if prep[i].item()==1 and result[i]==-1:
            max = -100
            result[i] = 0
            for k in self.res.keys():
                if self.res[k][i][1] > 0 and self.res[k][i][1] - self.res[k][i][0] > max:  # 对于正数的结果选差值最大的,结果0.6455
                    max = self.res[k][i][1] - self.res[k][i][0]
                    result[i] = k

        return result

is_support = torch.cuda.is_available()
if is_support:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

test_data = eval_dataset.MyDataset1(resize=224)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=True)

# model = NewModel('./populations1/models_indi2805/')
model = NewModel('./model_fine/')
model.to(device)

correct_sample = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        batch_x = data['data']
        batch_y = data['label']
        batch_x = Variable(batch_x).float().to(device)
        batch_y = Variable(batch_y).long().to(device)
        # print("inputs", batch_x.data.size(), "labels", batch_y.data)
        out = model(batch_x)

        correct_sample += (torch.Tensor(out).cpu() == batch_y.cpu()).sum().item()
        total += batch_y.size(0)
    acc = correct_sample / total
    print(acc)