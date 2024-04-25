#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
import sys
import typing

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(".")
import delay
import delay.agents
import delay.core

tf.disable_eager_execution()
keras = tf.keras
if typing.TYPE_CHECKING:
    from typing import *

print('Python', platform.python_version())
print('Tensorflow', tf.VERSION)
print('Keras', keras.__version__)

# Configuration

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

H = 3600
D = 24 * H

FIT_PREDICT_KWARGS = dict(
    batch_size=10000,
    verbose=0,
    callbacks=None,
)

DATA_PATH = '../../../../data/liuqm/cvr/CriteoCVR_old_base.feather'

COLUMN_CONFIG = delay.core.ColumnConfig(
    click_ts='click_timestamp',
    convert_ts='convert_timestamp',
    features={
        # name: (shape, dtype, categorical_size + 1, embedding_size)
        'int_1': ((), np.int8, 29, 8),
        'int_2': ((), np.int8, 13, 8),
        'int_3': ((), np.int16, 128, 16),
        'int_4': ((), np.int8, 41, 8),
        'int_5': ((), np.int16, 128, 16),
        'int_6': ((), np.int8, 24, 8),
        'int_7': ((), np.int16, 376, 32),
        'int_8': ((), np.int16, 284, 32),
        'cate_1': ((), np.int64, 35243, 512),
        'cate_2': ((), np.int16, 4054, 128),
        'cate_3': ((), np.int16, 11686, 256),
        'cate_4': ((), np.int16, 8081, 256),
        'cate_5': ((), np.int16, 3073, 64),
        'cate_6': ((), np.int16, 12893, 256),
        'cate_7': ((), np.int16, 9273, 256),
        'cate_8': ((), np.int8, 12, 16),
        'cate_9': ((), np.int16, 7186, 256),
        # 'int_1': ((), np.int8, 18, 8),
        # 'int_2': ((), np.int8, 14, 8),
        # 'int_3': ((), np.int16, 21, 16),
        # 'int_4': ((), np.int8, 21, 16),
        # 'int_5': ((), np.int16, 21, 16),
        # 'int_6': ((), np.int8, 21, 16),
        # 'int_7': ((), np.int16, 39, 16),
        # 'int_8': ((), np.int16, 26, 16),
        # 'cate_1': ((), np.int64, 35243, 512),
        # 'cate_2': ((), np.int16, 4054, 128),
        # 'cate_3': ((), np.int16, 11686, 256),
        # 'cate_4': ((), np.int16, 8081, 256),
        # 'cate_5': ((), np.int16, 3073, 64),
        # 'cate_6': ((), np.int16, 12893, 256),
        # 'cate_7': ((), np.int16, 9273, 256),
        # 'cate_8': ((), np.int8, 12, 16),
        # 'cate_9': ((), np.int16, 7186, 256),
    },
    # 用来计算 bias 的, name: categorical_size + 1
    bias_indexes={'campaignId': 13033},
    other_embedding_size=8,
)

MIN_TS, MAX_TS, STEP_TS, EVAL_TS = 0, 60 * D, 1 * H, 30 * D


def intermediate_layers_config_fn():
    return [
        (keras.layers.Dense, dict(units=128,
                                  activation=keras.layers.LeakyReLU(),
                                  kernel_regularizer=keras.regularizers.L1L2(l2=1e-6),
                                  name='hidden_1')
         ),
        (keras.layers.Dense, dict(units=128,
                                  activation=keras.layers.LeakyReLU(),
                                  kernel_regularizer=keras.regularizers.L1L2(l2=1e-6),
                                  name='hidden_2')
         ),
    ]


# Load data

data_provider = delay.core.DataProvider(df=pd.read_feather(DATA_PATH), cc=COLUMN_CONFIG,
                                        fast_index=(MIN_TS, MAX_TS, STEP_TS))
print(len(data_provider.df), data_provider.df[COLUMN_CONFIG.convert_ts].notnull().mean())


# Experiments

def run_exp(method: 'delay.core.MethodABC'):
    if isinstance(method, delay.core.Methods.FTP):
        agent_class = delay.agents.FTPAgent
    else:
        agent_class = delay.agents.SimpleDNNAgent
    agent = agent_class(method=method,
                        data_provider=data_provider,
                        intermediate_layers_config_fn=intermediate_layers_config_fn,
                        fit_predict_kwargs=FIT_PREDICT_KWARGS)
    result = delay.core.run_streaming(agent, min_ts=MIN_TS, max_ts=MAX_TS, step_ts=STEP_TS, eval_ts=EVAL_TS)
    print(method.description, result)
    return {'method': method.description, **result}


result_df = pd.DataFrame()

# Prophet

result_df = result_df.append(run_exp(delay.core.Methods.FNW(
    description='FNW',
)), ignore_index=True)

result_df = result_df.append(run_exp(delay.core.Methods.FNC(
    description='FNC',
)), ignore_index=True)

print(result_df)
log = open("a.txt", "a", encoding="utf-8")

print(result_df, file=log)
log.close()
