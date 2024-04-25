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
import glob
eval_acc = []
def evaluate(model,loader):   #计算每次训练后的准确率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = len(loader.dataset)
    for data in loader:
        batch_x = data['data']
        batch_y = data['label']
        batch_x = Variable(batch_x).float().to(device)
        batch_y = Variable(batch_y).long().to(device)
        # print("inputs", batch_x.data.size(), "labels", batch_y.data)
        out = model(batch_x)
        out = out.softmax(dim=1)
        _, pred = torch.max(out, 1)
        # print(pred)
        correct += torch.eq(pred,batch_y).sum().float().item()

    return correct/total


def train(datakind):
    # Step 0:查看torch版本、设置device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 36

    dataset = train_dataset.MyDataset(resize=224, datakind=datakind)#这里改图片size]

    train_taskset, val_taskset = train_test_split(dataset, train_size=0.8)
    train_loader = torch.utils.data.DataLoader(train_taskset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_taskset, batch_size=batch_size, shuffle=True)

    # Step 2: 初始化模型
    #model = models.resnet18()
    model = resnet18(pretrained=True,num_classes=2)
    # Step 3:设置损失函数
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # Step 4:选择优化器
    LR = 0.001
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    # Step 5:设置学习率下降策略
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Step 6:训练网络
    model.train()
    model.to(device)
    MAX_EPOCH = 20 # 设置epoch=20

    best_acc, best_epoch = 0, 0  # 输出验证集中准确率最高的轮次和准确率
    train_list, val_List = [], []  # 创建列表保存每一次的acc，用来最后的画图

    for epoch in range(MAX_EPOCH):
        loss_log = 0
        total_sample = 0
        train_correct_sample = 0
        for data in train_loader:
            input = data['data']
            label = data['label']
            # print(label)
            input = Variable(input).float().to(device)
            label = Variable(label).long().to(device)
            # ===================forward=====================
            output = model(input)
            #print(output)
            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            output=output.softmax(dim=1)
            _, predicted_label = torch.max(output, 1)#按行求最大值，并返回最大值的索引
            total_sample += label.size(0)
            train_correct_sample += (predicted_label == label).cpu().sum().numpy()
            loss_log += loss.item()

        model = model.cuda()

        train_list.append(train_correct_sample / total_sample)
        val_acc = evaluate(model, val_loader)
        print('val_acc=', val_acc)
        val_List.append((val_acc))
        # 打印信息
        print('epoch [{}/{}], loss:{:.4f}, acc:{:.4f}'
              .format(epoch + 1, MAX_EPOCH, loss_log / total_sample,train_correct_sample / total_sample))
        if val_acc > best_acc:  # 判断每次在验证集上的准确率是否为最大
            best_epoch = epoch
            best_acc = val_acc
            torch.save(model.state_dict(), "/tmp/code/gacnn/model/model_{}.pth".format(datakind))  # 保存验证集上最大的准确率

        scheduler.step()  # 更新学习率
        print('===========================分割线===========================')
        print('best acc:', best_acc, 'best_epoch:', best_epoch)
    print('train finish!')



def test(datakind):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = test_dataset.MyDataset2(resize=224, datakind=datakind)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    model = resnet18(num_classes=2)
    #print(model)

    modelpath = r"/tmp/code/gacnn/model/model_" + datakind + ".pth"

    # Step 3：加载训练好的权重
    trained_weight = torch.load(modelpath)
    model.load_state_dict(trained_weight)

    model.to(device)
    model.eval()
    correct_sample = 0
    total_sample = len(test_loader.dataset)

    with torch.no_grad():
        for data in test_loader:
            batch_x = data['data']
            batch_y = data['label']
            batch_x = Variable(batch_x).float().to(device)
            batch_y = Variable(batch_y).long().to(device)
            #print("inputs", batch_x.data.size(), "labels", batch_y.data)
            out = model(batch_x)
            out = out.softmax(dim=1)

            _, pred = torch.max(out, 1)
            #print(pred)
            correct_sample+= torch.eq(pred,batch_y).sum().float().item()
    print(correct_sample)
    print('Test:{} , Acc: {}'.format(datakind, correct_sample / total_sample))


def combine_and_eval(cur_order,task_num,order):
    is_support = torch.cuda.is_available()
    if is_support:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    test_data = eval_dataset.MyDataset1(resize=224,order=order,task_num=task_num)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    from newmodel import NewModel

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
    print(os.getcwd())
    directory = os.getcwd()
    # 获取所有文件
    files = glob.glob(directory + "/*")
    # 输出所有文件名
    for file in files:
        print(file)
    path = '/tmp/code/gacnn/datasets1/'
    dir = os.listdir(path)
    print(len(dir))
    # k=0
    # for i in sorted(dir):
    #     if i!='test_0':
    #         os.rename(path+i,path+str(k))
    #         k=k+1
    #order=random.sample(range(0, 100), 100)#任务顺序
    order=[5, 24, 92, 35, 63, 56, 82, 15, 6, 27, 23, 79, 31, 69, 49, 60, 89, 7, 93, 33, 12, 91, 9, 94, 47, 86, 74, 39, 13, 84,
     72, 70, 77, 83, 87, 17, 48, 68, 75, 90, 59, 19, 42, 21, 14, 76, 66, 57, 95, 54, 11, 98, 53, 38, 65, 37, 8, 40, 20,
     62, 4, 99, 16, 96, 61, 32, 78, 50, 29, 18, 2, 0, 58, 44, 30, 88, 3, 52, 73, 36, 43, 28, 1, 67, 10, 41, 22, 97, 55,
     25, 46, 64, 34, 45, 81, 85, 71, 80, 51, 26]
    print(order)
    task_num=0
    for k in range(0,20):

        task_num=task_num+1
        train(str(order[k]))
        print("train finish {}".format(order[k]))
        #test(str(order[k]))
        print("test finish {}".format(order[k]))
        #combine_and_eval(str(order[k]),task_num,order)

    print(eval_acc)
        # 画图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 12))
    eval_acc = np.array(eval_acc)
    t=range(eval_acc.size)
    ax1.plot(t, eval_acc, color='r', linestyle="-", marker='*', label='True',markersize=4.)

    eval_acc = np.array(eval_acc)
    t1=range(eval_acc.size)
    ax2.plot(t1, eval_acc, color='g', linestyle="-", marker='*', label='True',markersize=2.)
    plt.grid()
    plt.show()
   # plt.savefig('./eval_acc_.png')
    print(np.mean(eval_acc))

if __name__ == '__main__':
    main()


