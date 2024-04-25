# -*- coding: utf-8 -*-
import torchvision
from distance import getdistance
from resnet import resnet18
import torch
import os

from PIL import Image


def load_model(path1,path2,device):

    model0 = resnet18(num_classes=2)
    #print(model)
    model1 = resnet18(num_classes=2)
    # 加载训练好的权重
    trained_weight0 = torch.load(path1)
    model0.load_state_dict(trained_weight0)

    trained_weight1 = torch.load(path2)
    model1.load_state_dict(trained_weight1)
    # model0 = myvgg.get_trained_vgg(path=path1)
    # model1 = myvgg.get_trained_vgg(path=path2)


    model0 = model0.to(device)
    model1 = model1.to(device)
    model0.eval()
    model1.eval()
    return model0,model1

def get_testimg(device,path=r"./datasets/test/0.png",):
    img = Image.open(path)
    trans = []
    resize = 224
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
        # trans.append(torchvision.transforms.CenterCrop(size=resize))  # 中心剪裁
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)
    img = transform(img).unsqueeze(0).to(device)
    #print(img.shape)
    return img


def getindex(dis,thre=0.7):
    res=-1
    for i in range(len(dis)):
        if dis[i]<thre:
            #print(dis[i])
            res= i
            #print(res)
            break
    return res



def main():
    index_record=[]
    path = './model_fine/'
    dir = os.listdir(path)
    #print(len(dir))
    for i in sorted(dir):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if(i!='model_train_7.pth'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model1,model2=load_model(path1=r"./model_fine/{}".format(i),path2=r"./populations1/models_indi2805/model_{}.pth".format(i[12]),device=device)
            img=get_testimg(path=r"./datasets/test/0.png",device=device)
            out1=model1.forward_per_layer(img)
            #print(out1[1].shape)
            out2=model2.forward_per_layer(img)
            #print(out1)
            dis=getdistance(out1[1],out2[1])
            # index是最后一层可以复用的层数
            index=getindex(dis,thre=0.8)
            index_record.append(index)
            print(i,dis)
            print(index)
    print(index_record)

if __name__ == '__main__':

    main()