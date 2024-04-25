import torch
import torch.nn.functional as F
path='./model/model_train_apple.pth'

para=torch.load(path)

#print(para)
print(len(para))

path1 = './model/model_train_apple.pth'

para1 = torch.load(path1)

path2 = './model/model_train_baby.pth'
para2 = torch.load(path2)
res = []
print(para1['bn1.weight'].shape)
w1='layer1.0.conv1.weight'
w2='layer4.1.conv1.weight'
feature1 = para1[w2].view(para1[w2].shape[0], -1)  # 将特征转换为N*(C*W*H)，即两维
#print(feature1.shape)
#print(feature1)
feature2 = para2[w2].view(para2[w2].shape[0], -1)
#print(feature2)
feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一�?
print(feature1)
print(feature1.shape)
feature2 = F.normalize(feature2)
print(feature2)
distance = feature1.mm(feature2.t())  # 计算余弦相似�?
print(distance.item())