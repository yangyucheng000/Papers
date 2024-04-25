import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DATA_DIR = '../../../../data/liuqm/cvr/'

def read_feather_or_csv(csv_path, *args, **kwargs):
    feather_path = csv_path + '.feather'
    if os.path.exists(feather_path):
        print('Loading ' + feather_path)
        _df = pd.read_feather(feather_path)
    else:
        print('Loading ' + csv_path)
        _df = pd.read_csv(csv_path, *args, **kwargs)
        print('Dumping to feather ' + csv_path + '.feather')
        _df.to_feather(csv_path + '.feather')
    return _df

int_features = ['int_'+str(i) for i in range(1, 9)]
cate_features = ['cate_'+str(i) for i in range(1,10)]

df = read_feather_or_csv(
    DATA_DIR+'data.txt',
    sep='\t',
    names=['click_timestamp', 'convert_timestamp']+int_features+cate_features,
    dtype={
        'click_timestamp': np.int64,
        'convert_timestamp': 'Int64',
        **{col: 'Int64' for col in int_features},
        **{col: 'category' for col in cate_features},
    }
)

df = df.drop(index=df[df['convert_timestamp'] - df['click_timestamp'] < 0].index).reset_index(drop=True)

df = df[df['click_timestamp']<60*24*3600].sort_values('click_timestamp').reset_index(drop=True)
df[int_features] = df[int_features].fillna(-1)
df_int_test = df[int_features].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
#num_bin_size = (64, 16, 128, 64, 128, 64, 512, 512)
#pd.cut(df_int_test['int_1'], bins=[j/(num_bin_size[0]-1) for j in range(num_bin_size[0])], labels=[str(j) for j in range(num_bin_size[0] - 1)])
num_bin_size = (64, 16, 128, 64, 128, 64, 512, 512)
for i in range(1,9):
    df_int_test['int_{}'.format(i)] = pd.cut(df_int_test['int_{}'.format(i)],
                                             bins=[j/(num_bin_size[i-1]-1) for j in range(num_bin_size[i-1])],
                                             labels=[str(j) for j in range(num_bin_size[i-1] - 1)]
                                            )
df[int_features] = df_int_test

long_tail = {}
min_count_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
for col in tqdm(int_features + cate_features):
    counts = df[col].value_counts(dropna=False)
    long_tail[col] = [(counts<=c).sum() for c in min_count_index] + [len(counts)]

min_count = 10
df2 = df.copy()
df2['campaignId'] = df2['cate_3'].copy()

for col in tqdm(cate_features):
    df2[col].cat.add_categories('<UNK>', inplace=True)
    counts = df2[col].value_counts(dropna=False)
    df2[col][df2[col].isin(counts[counts<min_count].index)] = '<UNK>'
    
for col in tqdm(int_features + cate_features + ['campaignId']):
    df2[col].cat.remove_unused_categories(inplace=True)
    df2[col] = df2[col].cat.codes + 1

df2.to_feather(DATA_DIR + '/CriteoCVR_old_base.feather')