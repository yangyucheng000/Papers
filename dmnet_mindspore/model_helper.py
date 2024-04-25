import mindspore.nn as nn
import mindspore.ops as ops

def split(x, channels):
    return x.split(channels, 1)

def Flatten(x):
    return x.view((ops.shape(x)[0], -1))

def conv_pooling(in_channels, out_channels, pading=1):
    return nn.SequentialCell([
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='pad', padding=pading),
                        nn.BatchNorm2d(out_channels, momentum=1.0, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                        ])

def conv(in_channels, out_channels, pading=1):
    return nn.SequentialCell([
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='pad', padding=pading),
                        nn.BatchNorm2d(out_channels, momentum=1.0, affine=True),
                        nn.ReLU()
                        ])

def myconvpad(in_channels, out_channels, pading=(1, 1, 1 ,1)):
    return nn.SequentialCell([
                        nn.ReflectionPad2d(padding=pading),#外层填充
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='valid'),#卷积
                        nn.BatchNorm2d(out_channels, momentum=1.0, affine=True),#归一化
                        nn.ReLU()#激活函数
                        ])

def myconvpad_pooling(in_channels, out_channels, pading=(1, 1, 1 ,1)):
    return nn.SequentialCell([
                        nn.ReflectionPad2d(padding=pading),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='valid'),
                        nn.BatchNorm2d(out_channels, momentum=1.0, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                        ])

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    return ops.pow(x - y, 2).sum(2)