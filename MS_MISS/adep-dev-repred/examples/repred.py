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
import sklearn.metrics as sklearn_metrics
import tensorflow as tf

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

os.environ['CUDA_VISIBLE_DEVICES'] = GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.compat.v1.disable_eager_execution()
keras = tf.keras
kb = keras.backend
print(tf.version.VERSION, keras.__version__)
assert alib.version == ALIB_VERSION, 'alib: {} != {}'.format(alib.version, ALIB_VERSION)
assert delay.version == DELAY_VERSION, 'delay: {} != {}'.format(delay.version, DELAY_VERSION)
assert repred_lib.version == REPRED_VERSION, 'repred: {} != {}'.format(repred_lib.version, REPRED_VERSION)

COLUMN_CONFIG = delay.core.ColumnConfig(
    click_ts='click_timestamp',
    convert_ts='convert_timestamp',
    features={
        # name: (shape, dtype, categorical_size + 1, embedding_size)
        # 'int_1': ((), np.int8, 34, 8),
        # 'int_2': ((), np.int8, 21, 8),
        # 'int_3': ((), np.int8, 58, 8),
        # 'int_4': ((), np.int8, 42, 8),
        # 'int_5': ((), np.int8, 58, 8),
        # 'int_6': ((), np.int8, 47, 8),
        # 'int_7': ((), np.int16, 205, 8),
        # 'int_8': ((), np.int8, 82, 8),
        'cate_1': ((), np.int32, 35492, 8),
        # 'cate_2': ((), np.int16, 4068, 8),
        'cate_3': ((), np.int16, 11724, 8),
        'cate_4': ((), np.int16, 8130, 8),
        # 'cate_5': ((), np.int16, 3082, 8),
        'cate_6': ((), np.int16, 13027, 8),
        'cate_7': ((), np.int16, 9327, 8),
        # 'cate_8': ((), np.int8, 12, 8),
        'cate_9': ((), np.int16, 7228, 8),
    },
    # 用来计算 bias 的, name: categorical_size + 1
    bias_indexes={'campaignId': 13074},
    other_embedding_size=8,
)
HIDDEN_SIZES = [256, 32, 1]

H = 3600
D = 24 * H
MIN_TS, MAX_TS, STEP_TS, EVAL_TS = 0, 60 * D, 1 * H, 40 * D

FIT_PREDICT_KWARGS = dict(
    batch_size=10000,
    verbose=0,
    callbacks=None,
)

ie_list = [
    alib.model.InputExtension(col, *col_args) for col, col_args in COLUMN_CONFIG.features.items()
]


def build_ww(aw: 'str', lw: 'str') -> 'Tuple[repred_lib.Methods.WaitWindow, repred_lib.DnnModel]':
    name = f'WW_{aw}_{lw}'
    return (
        repred_lib.Methods.WaitWindow(
            name=name,
            aging_window=repred_lib.str_to_ts(aw),
            label_window=repred_lib.str_to_ts(lw),
        ),
        repred_lib.DnnModel(
            ies=ie_list,
            hidden_sizes=HIDDEN_SIZES,
            prefix=f'{name}/',
        ),
    )


def build_ww_ae(aw: 'str',
                lw: 'str',
                wws: 'Mapping[str, repred_lib.DnnModel]',
                ) -> 'Tuple[repred_lib.Methods.WaitWindowAE, repred_lib.DnnModel]':
    name = f'WW_{aw}_{lw}_AE_' + '_'.join(wws.keys())
    # noinspection PyArgumentList
    _method = repred_lib.Methods.WaitWindowAE(
        name=name,
        aging_window=repred_lib.str_to_ts(aw),
        label_window=repred_lib.str_to_ts(lw),
        ae_windows=[repred_lib.str_to_ts(w) for w in wws.keys()],
    )
    _model = repred_lib.DnnModel(
        ies=ie_list,
        hidden_sizes=HIDDEN_SIZES,
        prefix=f'{name}/',
        auxiliary_models=list(wws.values())
    )
    return _method, _model


_max_lw = 15
executors: 'List[Tuple[repred_lib.Methods.WaitWindow, repred_lib.DnnModel]]' = []
if 'base' in EXP_GROUPS:
    executors.extend([
        build_ww('00d', '31d'),  # Prophet
        *[build_ww(f'{k:02}d', f'{k:02}d') for k in range(1, 15)],  # Waiting, no (0,0)
    ])
if 'aging' in EXP_GROUPS:
    executors.extend([
        *[build_ww(f'{k:02}d', f'{_max_lw:02}d') for k in range(1, 2)],  # aging, skip (_max_lw,_max_lw)
    ])
if 'label' in EXP_GROUPS:
    executors.extend([
        *[build_ww(f'{0:02}d', f'{k:02}d') for k in range(1, _max_lw)],  # label, no (0,0), skip (0, _max_lw)
    ])
if 'more' in EXP_GROUPS:
    _more = 1
    executors.extend([
        *[build_ww(f'{k:02}d', f'{k + _more:02}d') for k in range(0, _max_lw + 1)],  # 1 more day
    ])
if 'more_ae1d' in EXP_GROUPS:
    fast_method, fast_model = build_ww('01d', '01d')
    executors.extend([
        (fast_method, fast_model),
        *[build_ww_ae(f'{k:02}d', f'{k + 1:02}d', {'01d': fast_model}) for k in range(0, _max_lw + 1)]
    ])
if 'more_ae1h' in EXP_GROUPS:
    fast_method, fast_model = build_ww('01h', '01h')
    executors.extend([
        (fast_method, fast_model),
        *[build_ww_ae(f'{k:02}d', f'{k + 1:02}d', {'01h': fast_model}) for k in range(0, _max_lw + 1)]
    ])

print('Methods:')
_method_set: 'Set[str]' = set()
for method, _ in executors:
    assert method.name not in _method_set, 'Duplicated method name: ' + method.name
    _method_set.add(method.name)
    print(method.name, method.identifier_versioning)
print('Total: ', len(executors))

data_provider = delay.core.DataProvider(df=pd.read_feather(DATA_PATH), cc=COLUMN_CONFIG,
                                        fast_index=(MIN_TS, MAX_TS, STEP_TS))
print('data_provider', len(data_provider.df), data_provider.df[COLUMN_CONFIG.convert_ts].notnull().mean())

for idx, repeat_id in enumerate(EXP_REPEATS):
    LAST_EPOCH = (idx == len(EXP_REPEATS) - 1)
    FIRST_EPOCH = (idx == 0)
    exp_id = f'{EXP_VERSION}_{",".join(EXP_GROUPS)}_{repeat_id}'
    # kb.clear_session()
    for ie in ie_list:
        ie.reset()

    for method, model in executors:
        data_provider.setup_write_back_col(col=method.name, fill_value=np.nan, dtype='float32')
        if FIRST_EPOCH:
            model.build(freeze_layers_after_build=True)
        # keras.utils.plot_model(model.ms, show_shapes=True, rankdir="LR")
        # model.mt.summary()

    pbar = tqdm(range(MIN_TS, EVAL_TS, STEP_TS), desc=f'Timeline({exp_id})')
    mbar = tqdm(total=len(executors))
    for ts_now in pbar:
        now_str = repred_lib.ts_to_str(ts_now)
        pbar.set_postfix({'now': now_str})

        eval_start = ts_now
        eval_end = eval_start + STEP_TS
        # if eval_start >= EVAL_TS:
        #     assert eval_end <= MAX_TS, f'eval_end={eval_end}'
        #     mbar.reset()
        #     mbar.set_description('Predict')
        #     for method, model in executors:
        #         df_serving = data_provider.serving_data(click_ts_start=ts_now, click_ts_end=ts_now + STEP_TS)
        #         y_pred = model.predict(df=df_serving, **FIT_PREDICT_KWARGS)
        #         data_provider.serving_write_back(
        #             index=df_serving.index.values, column_indexer=method.name, values=y_pred
        #         )
        #         mbar.update()

        mbar.reset()
        mbar.set_description('Train')
        for method, model in executors:
            train_start = ts_now - method.aging_window
            train_end = train_start + STEP_TS
            assert train_end <= eval_end, f'Current train_end={train_end}, but last eval_end={eval_end}'
            if train_start >= MIN_TS:
                assert train_end <= MAX_TS, f'train_end={train_end}'
                df_train = data_provider.get_unified_window(click_ts_start=train_start, click_ts_end=train_end,
                                                            ts_win=method.label_window)
                model.fit(df=df_train, y=COLUMN_CONFIG.label, shuffle=True, **FIT_PREDICT_KWARGS)
            mbar.update()
    mbar.set_description('')
    mbar.close()

    mbar = tqdm(total=len(executors))
    mbar.reset()
    mbar.set_description('Predict')
    for method, model in executors:
        df_serving = data_provider.serving_data(click_ts_start=EVAL_TS, click_ts_end=MAX_TS)
        y_pred = model.predict(df=df_serving, **FIT_PREDICT_KWARGS)
        data_provider.serving_write_back(
            index=df_serving.index.values, column_indexer=method.name, values=y_pred
        )
        mbar.update()
    mbar.set_description('')
    mbar.close()

    df_eval = data_provider.serving_data(click_ts_start=EVAL_TS, click_ts_end=MAX_TS)
    result_df = pd.DataFrame()

    mbar = tqdm(total=len(executors), desc=f'Result({exp_id})')
    for method, _ in executors:
        mbar.set_postfix({'method': method.name})
        y_true = df_eval[COLUMN_CONFIG.label].values
        y_pred = df_eval[method.name].values
        results = {
            'method': method.name,
            'identifier': method.identifier_versioning,
        }
        mbar.set_postfix({'computing': 'LogLoss'})
        results['LogLoss'] = sklearn_metrics.log_loss(y_true=y_true, y_pred=y_pred, labels=[0, 1])
        mbar.set_postfix({'computing': 'ROC-AUC'})
        results['ROC-AUC'] = sklearn_metrics.roc_auc_score(y_true=y_true, y_score=y_pred, labels=[0, 1])
        for bias_col, cat_size in COLUMN_CONFIG.bias_indexes.items():
            mbar.set_postfix({'computing': bias_col})
            sbm = alib.evaluation.StreamingBiasMetrics(size=cat_size)
            sbm.update(
                y_true=y_true,
                y_pred=y_pred,
                index=df_eval[bias_col].values,
            )
            results[f'Bias-{bias_col}'] = sbm.result(min_counts=100, min_sum=1)
            del sbm
        mbar.set_postfix({'computing': 'Done.'})
        result_df = result_df.append(results, ignore_index=True)
        mbar.update()
    mbar.close()
    print(result_df)
    result_df.to_feather(f'./Rst.{exp_id}.feather')
    # data_provider.df[[method.name for method, _ in executors]].to_feather(f'./dumps/pred.{exp_id}.feather')
