# -*- coding: utf-8 -*-
import train_dataset
import test_dataset
import eval_dataset
import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
from resnet import resnet18
from sklearn.model_selection import train_test_split
import os
import random
import matplotlib.pyplot as plt
import numpy as np

eval_acc = []


class NewModel(nn.Module):
    def __init__(self,task_num=None,order=None):
        super(NewModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.allmodel = {}
        self.trees={}
        self.totalkeys = {}
       # print(1)
        self.order=order[0:task_num]
        # self.order = random.sample(range(0, 10), 10)  # 随机生成顺序
        print(self.order)
        self.initmodels()
        # collections.OrderedDict()

    def initmodels(self):  # 修改合并网络及网络存放地址
        path = '/tmp/code/gacnn/populations_100/models_100t/'
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


def eval(cur_order,task_num,order):
    is_support = torch.cuda.is_available()
    if is_support:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    test_data = eval_dataset.MyDataset1(resize=224,order=order,task_num=task_num)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    #print(2)
    model = NewModel(task_num=task_num,order=order)

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

    # accuracy = float(correct) / total
    print("======================  Result  =============================")
    print(' Acc: {}'.format(correct_sample / total))
    eval_acc.append(correct_sample / total)


def main():
    import os
    global eval_acc
    eval_acc.clear()
    path = '/tmp/code/gacnn/datasets1/'
    dir = os.listdir(path)
    #print(len(dir))
    # k=0
    # for i in sorted(dir):
    #     if i!='test_0':
    #         os.rename(path+i,path+str(k))
    #         k=k+1
    #order=random.sample(range(0, 100), 100)#任务顺序
    order1=[11,15,16,20,23,24,27,31,32,35,40,41,44,5,50,56,6,63,66,71,74,79,82,87,88,92,93,95,96]
    order = [5, 24, 92, 35, 63, 56, 82, 15, 6, 27, 23, 79, 31, 69, 49, 60, 89, 7, 93, 33, 12, 91, 9, 94, 47, 86, 74, 39, 13, 84, 72, 70, 77, 83, 87, 17, 48, 68, 75, 90, 59, 19, 42, 21, 14, 76, 66, 57, 95, 54, 11, 98, 53, 38, 65, 37, 8, 40, 20, 62, 4, 99, 16, 96, 61, 32, 78, 50, 29, 18, 2, 0, 58, 44, 30, 88, 3, 52, 73, 36, 43, 28, 1, 67, 10, 41, 22, 97, 55, 25, 46, 64, 34, 45, 81, 85, 71, 80, 51, 26]
    print(len(order))
    task_num=0
    #order=order[:60]
    for k in range(0,len(order)):

        task_num=task_num+1
        eval(str(order[k]),task_num,order)

    print(eval_acc)
        # 画图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))
    eval_acc = np.array(eval_acc)
    t=range(eval_acc.size)
    ax1.plot(t, eval_acc, color='r', linestyle="-", marker='*', label='True')

    eval_acc = np.array(eval_acc)
    t1=range(eval_acc.size)
    ax2.plot(t1, eval_acc, color='g', linestyle="-", marker='*', label='True')

    plt.show()
    plt.savefig('/tmp/code/gacnn/eval_acc_100.png')
    print(np.mean(eval_acc))

if __name__ == '__main__':
    main()

