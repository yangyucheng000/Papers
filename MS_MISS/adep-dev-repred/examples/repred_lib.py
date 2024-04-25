# -*- coding: utf-8 -*-

import abc
from dataclasses import dataclass
import typing

import numpy as np
import pandas as pd

import alib

# keras = tf.keras
# kb = keras.backend
if typing.TYPE_CHECKING:
    from typing import *

version = '20220307-1215'




@dataclass
class MethodABC(abc.ABC):
    name: 'str'

    @property
    @abc.abstractmethod
    def identifier(self) -> 'str':
        pass

    @property
    def identifier_versioning(self) -> 'str':
        return self.identifier + '@' + version


class Methods(object):
    @dataclass
    class WaitWindow(MethodABC):
        aging_window: 'int'
        label_window: 'int'

        @property
        def identifier(self) -> 'str':
            return f'{type(self).__name__}({ts_to_str(self.aging_window)},{ts_to_str(self.label_window)})'

    @dataclass
    class WaitWindowAE(WaitWindow):
        ae_windows: 'Sequence[int]'

        @property
        def identifier(self) -> 'str':
            ae_str = ','.join([ts_to_str(w) for w in self.ae_windows])
            return super().identifier + f'+AE({ae_str})'

#
# class DnnModel(ModelABC):
#     hidden_sizes: 'Sequence[int]'
#     embedding_layers: 'Sequence[keras.layers.Layer]'
#     intermediate_layers: 'Sequence[keras.layers.Layer]'
#     _auxiliary_models: 'Sequence[DnnModel]'
#
#     def __init__(self,
#                  ies: 'Sequence[alib.model.InputExtension]',
#                  hidden_sizes: 'Sequence[int]',
#                  prefix: 'str' = '',
#                  auxiliary_models: 'Sequence[DnnModel]' = None
#                  ):
#         self.ies = ies
#         self.hidden_sizes = hidden_sizes
#         self.prefix = prefix
#         self._auxiliary_models = auxiliary_models or []
#
#     def build(self, freeze_layers_after_build=False):
#         embedding_layers = [
#             ie.to_embedding_layer(name_pattern=self.prefix + 'emb_{input_name}') for ie in self.ies
#         ]
#         ae_map_layers_s: 'List[List[keras.layers.Dense]]' = [
#             [
#                 keras.layers.Dense(
#                     units=am.embedding_layers[i].output_shape[-1],
#                     activation=None,
#                     use_bias=False,
#                     name=self.prefix + f'ae/' + am.embedding_layers[i].name
#                 ) for am in self._auxiliary_models
#             ] for i in range(len(self.ies))
#         ]
#         ae_sum_layers: 'List[keras.layers.Add]' = [
#             keras.layers.Add(
#                 name='ae_sum_' + self.ies[i].name
#             ) if len(ae_map_layers_s[i]) > 0 else None for i in range(len(self.ies))
#         ]
#         assert len(embedding_layers) == len(ae_map_layers_s) == len(ae_sum_layers), 'length not match.'
#         concat_embedding_layer = keras.layers.Concatenate(name=self.prefix + 'concat_emb')
#         if len(self.hidden_sizes) == 0:
#             hidden_layer_list = [
#                 keras.layers.Lambda(kb.sum, arguments=dict(axis=-1, keepdims=True), name=self.prefix + 'lr_sum')
#             ]
#         else:
#             assert self.hidden_sizes[-1] == 1, f'Invalid hidden sizes {self.hidden_sizes}'
#             hidden_layer_list = [
#                 keras.layers.Dense(
#                     units=hidden_size,
#                     activation=None if i == len(self.hidden_sizes) - 1 else keras.layers.LeakyReLU(
#                         name=self.prefix + f'LeakyReLU_{i}'),
#                     name=self.prefix + f'hidden_{i}'
#                 )
#                 for i, hidden_size in enumerate(self.hidden_sizes)]
#         sigmoid_layer = keras.layers.Activation('sigmoid', name=self.prefix + 'sigmoid')
#         out_squeeze_layer = keras.layers.Lambda(kb.squeeze, name=self.prefix + 'out_squeeze', arguments=dict(axis=-1))
#
#         self.embedding_layers = embedding_layers
#         self.intermediate_layers = hidden_layer_list
#
#         # net
#         ie_input_list = [ie.keras_input for ie in self.ies]
#         embeddings = [
#             layer(ie_input) for layer, ie_input in zip(embedding_layers, ie_input_list)
#         ]
#         ae_tensors_s: 'List[List[tf.Tensor]]' = [
#             [
#                 am.embedding_layers[i](ie_input_list[i]) for am in self._auxiliary_models
#             ] for i in range(len(self.ies))
#         ]
#         ensemble_embeddings = [
#             aes_l([emb] + [l(t) for t, l in zip(ae_ts, aem_ls)]) if len(ae_ts) > 0 else emb
#             for emb, ae_ts, aem_ls, aes_l in zip(embeddings, ae_tensors_s, ae_map_layers_s, ae_sum_layers)
#         ]
#         assert len(ensemble_embeddings) == len(embeddings), 'embeddings length not match.'
#         net = concat_embedding_layer(ensemble_embeddings)
#         for layer in hidden_layer_list:
#             net = layer(net)
#         out_prob = out_squeeze_layer(sigmoid_layer(net))
#
#         # Serving
#         self.ms = keras.Model(ie_input_list, out_prob)
#
#         # Training
#         self.mt = keras.Model(ie_input_list, net)
#         self.mt.compile(
#             optimizer=keras.optimizers.Adam(),
#             loss=keras.losses.BinaryCrossentropy(name='LogLoss', from_logits=True)
#         )
#
#         # Freeze layers
#         if freeze_layers_after_build:
#             for layer in self.embedding_layers:
#                 layer.trainable = False
#             for layer in self.intermediate_layers:
#                 layer.trainable = False
#
#     def init_layer(self, layers, t=0):
#         if isinstance(layers, list):
#             for layer in layers:
#                 session = kb.get_session()
#                 weights_initializer = tf.variables_initializer(layer.weights)
#                 session.run(weights_initializer)
#         else:
#             session = kb.get_session()
#             # print(session.run(layers.weights))
#             # origin_weights = tf.Variable(tf.random_normal(shape=layers.weights[0].shape, mean=0, stddev=0.001),
#             #                              trainable=False, name='v')
#             origin_weights = tf.random_normal(layers.weights[0].shape, mean=0, stddev=0.01)
#             # origin_weights.detach()
#             # weights_initializer = tf.variables_initializer([origin_weights])
#             # session.run(weights_initializer)
#             # update_o = tf.assign(origin_weights, t * layers.weights[0] / (t+1))
#             # session.run(update_o)
#             # origin_weights = layers.weights[0]
#             # c = tf.constant('v')
#             print("origin:", origin_weights.name, session.run(origin_weights))
#             weights_initializer = tf.variables_initializer(layers.weights)
#             # session.run(weights_initializer)
#             print("init:", session.run(layers.weights))
#             new_value = t * origin_weights / (t + 1) + 1 * layers.weights[0] / (t+1)
#             update = tf.assign(layers.weights[0], new_value)
#             session.run(update)
#             print("final:")
#             print(session.run(layers.weights))
#

def ts_to_str(ts: 'int') -> 'str':
    return f'{ts // (24 * 3600) :02}d{ts % (24 * 3600) // 3600 :02}h'


def str_to_ts(s: 'str') -> 'int':
    ts = 0
    if 'd' in s:
        elements = s.split('d', maxsplit=1)
        ts += int(elements[0], base=10) * 3600 * 24
        s = elements[1]
    if 'h' in s:
        elements = s.split('h', maxsplit=1)
        ts += int(elements[0], base=10) * 3600
        s = elements[1]
    assert len(s) == 0, 'parse error.'
    return ts
