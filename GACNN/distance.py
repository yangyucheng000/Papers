# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F



def getdistance(datalist1,datalist2):
    res=[]
    for i in range(len(datalist1)):
        feature1 = datalist1[i].view(datalist1[i].shape[0], -1)  # 灏嗙壒寰佽浆鎹负N*(C*W*H)锛屽嵆涓ょ淮
        #print(feature1)
        feature2 = datalist2[i].view(datalist2[i].shape[0], -1)
        feature1 = F.normalize(feature1)  # F.normalize鍙兘澶勭悊涓ょ淮鐨勬暟鎹紝L2褰掍竴鍖?
        feature2 = F.normalize(feature2)
        distance = feature1.mm(feature2.t())  # 璁＄畻浣欏鸡鐩镐技搴?
        #print('a:',distance)
        res.append(distance.item())
    #print(res)
    return res