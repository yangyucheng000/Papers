from __future__ import division

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
import math
import numpy as np


class GraphConvolution(nn.Cell):
    def __init__(self, in_features, out_features, dropout, edge_dropout, n_layers, activation, highway, bias=True):
        super(GraphConvolution, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.in_features = in_features
        self.n_layers = n_layers
        self.out_features = out_features
        self.edge_dropout = edge_dropout
        self.activation = activation
        self.highway = highway
        self.layer_1 = GraphConvolutionLayer(in_features,
                                             out_features,
                                             edge_dropout,
                                             activation,
                                             highway,
                                             bias)
        if n_layers == 2:
            self.layer_2 = GraphConvolutionLayer(in_features,
                                                 out_features,
                                                 edge_dropout,
                                                 activation,
                                                 highway,
                                                 bias)
        else:
            self.layer_2 = None
        assert (n_layers == 1 or n_layers == 2)

    def construct(self, inputs, adj):
        features = self.dropout(self.layer_1(inputs, adj))
        if self.layer_2:
            features = self.dropout(self.layer_2(features, adj))
        return features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ', ' \
               + 'activation=' + str(self.activation) + ', ' \
               + 'highway=' + str(self.highway) + ', ' \
               + 'layers=' + str(self.n_layers) + ', ' \
               + 'dropout=' + str(self.dropout.p) + ', ' \
               + 'edge_dropout=' + str(self.edge_dropout) + ')'


class GraphConvolutionLayer(nn.Cell):
    """
    From https://github.com/tkipf/pygcn.
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, edge_dropout, activation, highway, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        out_features = int(out_features)
        self.out_features = out_features
        self.edge_dropout = nn.Dropout(p=edge_dropout)
        self.highway = highway
        self.activation = activation
        self.weight = Parameter(Tensor(np.empty((3, in_features, out_features)), dtype=mindspore.float32))
        if bias:
            self.bias = Parameter(Tensor(np.empty((3, 1, out_features)), dtype=mindspore.float32))
        if highway != "":
            assert (in_features == out_features)
            self.weight_highway = Parameter(Tensor(np.empty((in_features, out_features)), dtype=mindspore.float32))
            self.bias_highway = Parameter(Tensor(np.empty((1, out_features)), dtype=mindspore.float32))

        else:
            self.bias = None
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.set_data(ops.uniform(self.weight.shape, Tensor(-stdv, mindspore.float32), Tensor(stdv, mindspore.float32)))
        # self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.set_data(ops.uniform(self.bias.shape, Tensor(-stdv, mindspore.float32), Tensor(stdv, mindspore.float32)))
            # self.bias.data.uniform_(-stdv, stdv)

    def construct(self, inputs, adj):
        features = self.edge_dropout(inputs)
        outputs = []
        for i in range(features.shape[1]):
            support = ops.bmm(
                features[:, i, :].unsqueeze(0).broadcast_to((self.weight.shape[0], *features[:, i, :].shape)),
                self.weight
            )
            if self.bias is not None:
                support += self.bias.expand_as(support)

            output = ops.mm(
                adj[:, i, :].swapaxes(1, 2).view(support.shape[0] * support.shape[1], -1).swapaxes(0, 1),
                support.view(support.shape[0] * support.shape[1], -1)
            )
            outputs.append(output)
        if self.activation == "leaky_relu":
            output = ops.leaky_relu(ops.stack(outputs, 1))
        elif self.activation == "relu":
            output = ops.relu(ops.stack(outputs, 1))
        elif self.activation == "tanh":
            output = ops.tanh(ops.stack(outputs, 1))
        elif self.activation == "sigmoid":
            output = ops.sigmoid(ops.stack(outputs, 1))
        else:
            assert (False)

        if self.highway != "":
            transform = []
            for i in range(features.shape[1]):
                transform_batch = ops.mm(features[:, i, :], self.weight_highway)
                transform_batch += self.bias_highway.expand_as(transform_batch)
                transform.append(transform_batch)
            if self.highway == "leaky_relu":
                transform = ops.leaky_relu(ops.stack(transform, 1))
            elif self.highway == "relu":
                transform = ops.relu(ops.stack(transform, 1))
            elif self.highway == "tanh":
                transform = ops.tanh(ops.stack(transform, 1))
            elif self.highway == "sigmoid":
                transform = ops.sigmoid(ops.stack(transform, 1))
            else:
                assert (False)
            carry = 1 - transform
            output = output * transform + features * carry
        return output
