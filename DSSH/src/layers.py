""" Some layers for build Cell """
import math
import warnings
from itertools import repeat
import collections.abc
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn


class Identity(nn.Cell):
    def __init__(self):
        super(Identity, self).__init__()

    def construct(self, x):
        return x


class DropPath(nn.Cell):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.rand = ops.UniformReal(seed=0)
        self.floor = ops.Floor()

    def construct(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = self.rand(shape)
        random_tensor = keep_prob + random_tensor # ops.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = self.floor(random_tensor)  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Cell):
    def __init__(self, in_features: int, hidden_features=None, out_features=None, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features, has_bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Dense(hidden_features, out_features, has_bias=True)
        self.drop = nn.Dropout(drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x