# -*- coding: utf-8 -*-
# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import mindspore
import numpy as np
from mindspore import Tensor
# from mindspore.ops import operations as ops
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import common


def st_grid_calculation(st_relevance_score, word_id_st_sent2wordlist, bbox_st_list, word_id_st_sent, st_list_bbox2word,
                        visu_scale, image_scale):
    batch_size = st_relevance_score.shape[0]
    dividend = image_scale // visu_scale
    activation_map = ops.Zeros()(Tensor([batch_size, visu_scale, visu_scale, 1]))
    for batch_i in range(batch_size):
        for ii in range(len(st_relevance_score[batch_i])):
            if not st_relevance_score[batch_i][ii] == 0:
                bbox_index = ops.NotEqual()(st_list_bbox2word[batch_i],
                                            (word_id_st_sent2wordlist[batch_i][ii] + 1)).nonzero()
                for jj in bbox_index:
                    x1, y1, x2, y2 = bbox_st_list[batch_i][jj.item()]
                    grid_xl = (x1 // dividend).int().item()
                    grid_xr = min((x2 // dividend + 1).int().item(), visu_scale - 1)
                    grid_yt = (y1 // dividend).int().item()
                    grid_yb = min((y2 // dividend + 1).int().item(), visu_scale - 1)
                    activation_map[batch_i, grid_yt:grid_yb, grid_xl:grid_xr] = st_relevance_score[batch_i][ii].item()
    # grid softmax
    for batch_i in range(batch_size):
        if not len(ops.NotEqual()(activation_map[batch_i], 0).nonzero()) == 0:
            tmp = activation_map[batch_i]
            tmp = tmp.reshape(-1, 1)
            tmp = nn.Softmax()(tmp * 9)
            tmp = tmp.reshape(visu_scale, visu_scale, -1)
            activation_map[batch_i] = tmp
    return activation_map


import torch


# # 通过PyTorch参数文件，打印PyTorch的参数文件里所有参数的参数名和shape，返回参数字典
# def pytorch_params(pth_file):
#     par_dict = torch.load(pth_file, map_location='cpu')
#     return par_dict
#     pt_params = {}
#     for name in par_dict:
#         parameter = par_dict[name]
#         print(name, parameter.numpy().shape)
#         pt_params[name] = parameter.numpy()
#     return pt_params
#
#
# # 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
# def mindspore_params(network):
#     ms_params = {}
#     for param in network.get_parameters():
#         name = param.name
#         value = param.data.asnumpy()
#         print(name, value.shape)
#         ms_params[name] = value
#     return ms_params
#


class ConvBatchNormReLU(nn.SequentialCell):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            leaky=False,
            relu=True,
            instance=False,
    ):
        super(ConvBatchNormReLU, self).__init__()
        self.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pad_mode='pad',
                padding=padding,
                dilation=dilation,
                has_bias=False
            )
        )
        if instance:
            self.append(
                nn.InstanceNorm2d(num_features=out_channels)
            )
        else:
            self.append(
                nn.BatchNorm2d(
                    out_channels, eps=1e-5, momentum=0.001, affine=True
                )
            )

        if leaky:
            self.append(nn.LeakyReLU(0.1))
        elif relu:
            self.append(nn.ReLU())

    def construct(self, x):
        return super(ConvBatchNormReLU, self).construct(x)


# mindspore
y = ops.zeros((1, 2, 3), dtype=mindspore.float32)
print(y.dtype)
print(y.dtype == mindspore.float32)
print(y.dtype is mindspore.float32)


# 对数据进行 L2 范数归一化
# X_norm = l2norm(X, dim=0)
# print("归一化后的数据：")
# print(X_norm)

# class MyUpsample2(nn.Cell):
#     def construct(self, x):
#         return ops.Reshape()(ops.Tile()(ops.ExpandDims()(ops.ExpandDims()(x, 3), 5), (1, 1, 1, 2, 1, 2)),
#                              (x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] * 2))
#
#
# class MyUpsample3(torch.nn.Module):
#     def forward(self, x):
#         return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2) * 2,
#                                                                               x.size(3) * 2)
#
#
# # 创建一个输入张量
# x = mindspore.Tensor(np.zeros((2, 2, 2, 2)))
#
# t_x = torch.zeros((2, 2, 2, 2))
#
# # 创建一个MyUpsample2类的实例
# upsample = MyUpsample2()
#
# # 对输入张量进行上采样
# y = upsample(x)
#
# # 输出结果
# print(y)
# print(y.shape)
#
# upsample2 = MyUpsample3()
#
# t_y = upsample2(t_x)
# print(t_y, t_y.shape)
