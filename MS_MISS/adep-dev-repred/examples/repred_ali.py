#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

sys.path.append('../adep-dev-repred/')
sys.path.append('../adep-dev-repred/examples')

import argparse
import os
import typing

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import gc
import sklearn.metrics as sklearn_metrics
# import tensorflow as tf

import mindspore
import mindspore as ms
from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
import mindspore.dataset as ds
from mindspore import ops
import pdb
from mindspore import Tensor

import alib
import delay
import delay.core
import repred_lib

if typing.TYPE_CHECKING:
    from typing import *

DATA_PATH = '../../adep-dev-repred/data-criteo/CriteoCVR-20211203.feather'
ALIB_VERSION = '20220308-2000'
DELAY_VERSION = '20220303-1600'
REPRED_VERSION = '20220307-1215'
EXP_VERSION = '13'

parser = argparse.ArgumentParser(description='RePred')
parser.add_argument('--gpu', required=True, type=str)
parser.add_argument('--egs', required=True, type=str)
parser.add_argument('--ers', required=True, type=str)
args = parser.parse_args()
GPU: 'str' = args.gpu
EXP_GROUPS: 'List[str]' = args.egs.split(',')
EXP_REPEATS: 'List[str]' = args.ers.split(',')

assert alib.version == ALIB_VERSION, 'alib: {} != {}'.format(alib.version, ALIB_VERSION)
assert delay.version == DELAY_VERSION, 'delay: {} != {}'.format(delay.version, DELAY_VERSION)
assert repred_lib.version == REPRED_VERSION, 'repred: {} != {}'.format(repred_lib.version, REPRED_VERSION)

COLUMN_CONFIG = delay.core.ColumnConfig(
    click_ts='click_timestamp',
    convert_ts='convert_timestamp',
    features={
        # name: (shape, dtype, categorical_size + 1, embedding_size)
        'int_1': ((), np.int8, 34, 8),
        'int_2': ((), np.int8, 21, 8),
        'int_3': ((), np.int8, 58, 8),
        'int_4': ((), np.int8, 42, 8),
        'int_5': ((), np.int8, 58, 8),
        'int_6': ((), np.int8, 47, 8),
        'int_7': ((), np.int16, 205, 8),
        'int_8': ((), np.int8, 82, 8),
        'cate_1': ((), np.int32, 35492, 8),
        'cate_2': ((), np.int16, 4068, 8),
        'cate_3': ((), np.int16, 11724, 8),
        'cate_4': ((), np.int16, 8130, 8),
        'cate_5': ((), np.int16, 3082, 8),
        'cate_6': ((), np.int16, 13027, 8),
        'cate_7': ((), np.int16, 9327, 8),
        'cate_8': ((), np.int8, 12, 8),
        'cate_9': ((), np.int16, 7228, 8),
    },
    # 用来计算 bias 的, name: categorical_size + 1
    bias_indexes={'campaignId': 13074},
    other_embedding_size=8,
)
HIDDEN_SIZES = [128, 32, 1]

H = 3600
D = 24 * H
MIN_TS, MAX_TS, STEP_TS, EVAL_TS = 0, 60 * D, 4 * H, 40 * D

FIT_PREDICT_KWARGS = dict(
    batch_size=10000,
    verbose=0,
    callbacks=None,
)


class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.Emb = nn.SequentialCell(
            nn.Embedding(35493, 8, False),
            nn.Flatten(),
        )
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(17 * 8, 256),
            nn.LeakyReLU(),
            nn.Dense(256, 128),
            nn.LeakyReLU(),
            nn.Dense(128, 5),
            nn.Sigmoid()
        )

    def construct(self, x):
        # print(x.shape)
        x = self.Emb(x)
        # print(x.shape)
        logits = self.dense_relu_sequential(x)
        return logits


def build_ww(aw: 'str', lw: 'str') -> 'repred_lib.Methods.WaitWindow':
    name = f'WW_{aw}_{lw}'
    return repred_lib.Methods.WaitWindow(
        name=name,
        aging_window=repred_lib.str_to_ts(aw),
        label_window=repred_lib.str_to_ts(lw),
    )


_max_lw = 15
executors: 'List[repred_lib.Methods.WaitWindow]' = []
if 'base' in EXP_GROUPS:
    executors.extend([
        # build_ww('00d', '31d'),  # Prophet
        *[build_ww(f'{k:02}d', f'{k:02}d') for k in [1, 7, 14, 21, 30]],  # Waiting, no (0,0)
    ])
if 'aging' in EXP_GROUPS:
    executors.extend([
        *[build_ww(f'{k:02}d', f'{_max_lw:02}d') for k in range(1, 2)],  # aging, skip (_max_lw,_max_lw)
    ])
if 'label' in EXP_GROUPS:
    executors.extend([
        *[build_ww(f'{0:02}d', f'{k:02}d') for k in range(1, _max_lw)],  # label, no (0,0), skip (0, _max_lw)
    ])

print('Methods:')
_method_set: 'Set[str]' = set()
print(executors)
for method in executors:
    assert method.name not in _method_set, 'Duplicated method name: ' + method.name
    _method_set.add(method.name)
    print(method.name, method.identifier_versioning)
print('Total: ', len(executors))

samples = pd.read_feather(DATA_PATH)
# samples = samples.sample(frac=1, replace=True)
# samples = samples[:100000]

data_provider = delay.core.DataProvider(df=pd.read_feather(DATA_PATH), cc=COLUMN_CONFIG,
                                        fast_index=(MIN_TS, MAX_TS, STEP_TS))
print('data_provider', len(data_provider.df), data_provider.df[COLUMN_CONFIG.convert_ts].notnull().mean())

model = Network()
loss_fn = nn.BCELoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.005)
pos_weight = [1.2, 1.1, 1.05, 1.03, 1]


def forward_fn(data, label, task_id):
    # pdb.set_trace()
    logits = model(data)
    # print(logits.shape)
    logits = logits[:, task_id]
    # print(logits.shape)
    label = label.reshape(-1)
    weight = label.asnumpy().view()
    pos = pos_weight[task_id]
    asign = lambda t: 1 if t == 0 else pos
    weight = Tensor(np.array(list(map(asign, weight))), ms.float32)
    # pweight = label.deepcopy()
    # pweight
    # logits = Tensor.from_numpy(logits.asnumpy().reshape(msbatch, -1))
    # print(logits, label)
    loss_fn = nn.BCELoss(weight=weight)
    loss = loss_fn(logits, label)
    # print(sklearn_metrics.roc_auc_score(y_true=label.asnumpy(), y_score=logits.asnumpy(), labels=[0, 1]))

    return loss, logits


# Get gradient function
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
msbatch = 10000


# Define function of one-step training
def train_step(data, label, task_id):
    (loss, _), grads = grad_fn(data, label, task_id)
    optimizer(grads)
    return loss


def train_loop(model, dataset, task_id):
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, data in enumerate(dataset.create_tuple_iterator()):
        d = []

        label = Tensor(data[-1].asnumpy().reshape(-1, 1), ms.float32)
        # print(label.shape)
        for t in data[:-1]:
            d.append(t.asnumpy().reshape(-1, 1))
        d = np.concatenate(d, axis=1)
        data = Tensor.from_numpy(d)
        # print(label)
        # print(data)

        loss = train_step(data, label, task_id)

        # if batch % 1 == 0:
        #     loss, current = loss.asnumpy(), batch
        #     print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


def test_loop(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    total_pred = []
    total_auc = 0
    total_nll = 0
    total_prauc = 0
    # pdb.set_trace()
    for data in dataset.create_tuple_iterator():
        d = []
        label = Tensor(data[-1].asnumpy().reshape(-1, 1), ms.float32)
        for t in data[:-1]:
            d.append(t.asnumpy().reshape(-1, 1))
        d = np.concatenate(d, axis=1)
        data = Tensor.from_numpy(d)
        data_num = len(data)
        pred = model(data)
        pred = ops.mul(pred, Tensor(np.array([0.1, 0.1, 0.2, 0.3, 0.3]).reshape(1, -1)))
        print("after_mul", pred.shape)
        pred = ops.sum(pred, dim=1)
        total_pred.append(pred.asnumpy().reshape(-1, 1))
        temp_auc = sklearn_metrics.roc_auc_score(y_true=label.asnumpy(), y_score=pred.asnumpy(), labels=[0, 1])
        temp_nll = sklearn_metrics.log_loss(y_true=label.asnumpy(), y_pred=pred.asnumpy(), labels=[0, 1])
        temp_prauc = sklearn_metrics.average_precision_score(y_true=label.asnumpy(), y_score=pred.asnumpy())
        total_auc += temp_auc * data_num
        total_nll += temp_nll * data_num
        total_prauc += temp_prauc * data_num
        total += len(data)
        # test_loss += loss_fn(pred, label).asnumpy()
        # correct += (pred.argmax(1) == label).asnumpy().sum()
    # test_loss /= num_batches

    # correct /= total
    # print(f"Test:  Avg loss: {test_loss:>8f} \n")
    return np.concatenate(total_pred, axis=0).reshape(-1, 1), total_nll, total_auc, total_prauc, total


for idx, repeat_id in enumerate(EXP_REPEATS):
    exp_id = f'{EXP_VERSION}_{",".join(EXP_GROUPS)}_{repeat_id}'
    # kb.clear_session()
    # for ie in ie_list:
    #     ie.reset()
    eval_total = 0
    eval_auc = 0
    eval_nll = 0
    eval_prauc = 0
    for method in executors:
        data_provider.setup_write_back_col(col=method.name, fill_value=np.nan, dtype='float32')

    pbar = tqdm(range(MIN_TS, MAX_TS, STEP_TS), desc=f'Timeline({exp_id})')
    mbar = tqdm(total=len(executors))
    for ts_now in pbar:
        now_str = repred_lib.ts_to_str(ts_now)
        pbar.set_postfix({'now': now_str})

        eval_start = ts_now
        eval_end = eval_start + STEP_TS

        if eval_start >= EVAL_TS:
            assert eval_end <= MAX_TS, f'eval_end={eval_end}'
            mbar.reset()
            mbar.set_description('Predict')
            for method in executors[:1]:
                df_serving_real = data_provider.serving_data(click_ts_start=ts_now, click_ts_end=ts_now + STEP_TS)
                co = ["int_1", "int_2", "int_3", "int_4", "int_5", "int_6", "int_7", "int_8", "cate_1",
                      "cate_2", "cate_3", "cate_4", "cate_5", "cate_6", "cate_7", "cate_8", "cate_9", "label"]
                df_serving = df_serving_real.loc[:,
                             ("int_1", "int_2", "int_3", "int_4", "int_5", "int_6", "int_7", "int_8", "cate_1",
                              "cate_2", "cate_3", "cate_4", "cate_5", "cate_6", "cate_7", "cate_8", "cate_9",
                              "cc__label")]
                df_serving = df_serving.to_dict(orient='list')
                df_serving = ds.NumpySlicesDataset(data=df_serving, column_names=co)
                # print(df_serving.get_col_names())
                df_serving = df_serving.batch(msbatch)

                y_pred, total_nll, total_auc, total_prauc, total = test_loop(model, df_serving, loss_fn)
                eval_total += total
                eval_nll += total_nll
                eval_auc += total_auc
                eval_prauc += total_prauc
                # y_pred = model.predict(df=df_serving, **FIT_PREDICT_KWARGS)

                data_provider.serving_write_back(
                    index=df_serving_real.index.values, column_indexer=method.name, values=y_pred
                )
                del df_serving
                gc.collect()
                mbar.update()

        mbar.reset()
        mbar.set_description('Train')

        for task_id, method in enumerate(executors):
            train_start = ts_now - method.aging_window
            train_end = train_start + STEP_TS
            # assert train_end <= eval_end, f'Current train_end={train_end}, but last eval_end={eval_end}'
            if train_start >= MIN_TS:
                assert train_end <= MAX_TS, f'train_end={train_end}'
                df_train = data_provider.get_fake_negative_vanilla(click_ts_start=train_start, click_ts_end=train_end,
                                                                   ts_now_start=ts_now, ts_now_end=ts_now + STEP_TS,
                                                                   ts_win=method.label_window)
                # print(df_train[0:1])
                # pdb.set_trace()
                co = ["int_1", "int_2", "int_3", "int_4", "int_5", "int_6", "int_7", "int_8", "cate_1",
                      "cate_2", "cate_3", "cate_4", "cate_5", "cate_6", "cate_7", "cate_8", "cate_9", "label"]
                df_train = df_train.loc[:,
                           ("int_1", "int_2", "int_3", "int_4", "int_5", "int_6", "int_7", "int_8", "cate_1",
                            "cate_2", "cate_3", "cate_4", "cate_5", "cate_6", "cate_7", "cate_8", "cate_9",
                            "cc__label")]
                df_train = df_train.to_dict(orient='list')
                # print(df_train)
                # df_train = df_train.values
                # df_train = df_train.fillna(0)
                df_train = ds.NumpySlicesDataset(data=df_train, column_names=co)
                # print(df_train.get_col_names())
                # print(df_train)
                df_train = df_train.batch(msbatch)
                for t in range(1):
                    print(f"Epoch {t + 1}\n-------------------------------")

                    train_loop(model, df_train, task_id)
                    # test_loop(model, test_dataset, loss_fn)
                print("Done!")

                # model.fit(df=df_train, y=COLUMN_CONFIG.label, shuffle=True, **FIT_PREDICT_KWARGS)

            mbar.update()
    mbar.set_description('')
    mbar.close()

    df_eval = data_provider.serving_data(click_ts_start=EVAL_TS, click_ts_end=MAX_TS)
    result_df = pd.DataFrame()

    mbar = tqdm(total=len(executors), desc=f'Result({exp_id})')
    for tid, method in enumerate(executors):
        mbar.set_postfix({'method': method.name})
        if tid == 0:
            y_true = df_eval[COLUMN_CONFIG.label].values
            y_pred = df_eval[method.name].values
            results = {
                'method': method.name,
                'identifier': method.identifier_versioning,
            }
            mbar.set_postfix({'computing': 'LogLoss'})
            results['LogLoss'] = eval_nll / eval_total
            # sklearn_metrics.log_loss(y_true=y_true, y_pred=y_pred, labels=[0, 1])
            mbar.set_postfix({'computing': 'ROC-AUC'})
            results['ROC-AUC'] = eval_auc / eval_total
            # sklearn_metrics.roc_auc_score(y_true=y_true, y_score=y_pred, labels=[0, 1])
            pdb.set_trace()
            results['PR-AUC'] = eval_prauc / eval_total
            # sklearn_metrics.average_precision_score(y_true=y_true, y_score=y_pred)
            mbar.set_postfix({'computing': 'Done.'})
            result_df = result_df.append(results, ignore_index=True)
        mbar.update()
    mbar.close()
    print(result_df)
    result_df.to_feather(f'./Rst.{exp_id}.feather')
    # data_provider.df[[method.name for method, _ in executors]].to_feather(f'./dumps/pred.{exp_id}.feather')
