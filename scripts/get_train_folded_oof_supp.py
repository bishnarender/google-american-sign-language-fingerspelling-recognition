#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import glob
from rapidfuzz.distance.DamerauLevenshtein_py import distance
import json
import sys


#TODO add argsparse

train = pd.read_csv('datamount/train_folded.csv')
train


# In[11]:


val_data_fns = [glob.glob(f'datamount/weights/cfg_1/fold{i}/val_data_seed*.pth') for i in [0,1,2,3]]
print('getting oofs from')
print(val_data_fns)


def get_score(phrase_gt, phrase_preds):
    
    N = np.array([len(p) for p in phrase_gt])
    D = np.array([distance(p1,p2) for p1,p2 in zip(phrase_gt,phrase_preds)])
    score = (N - D) / N   
    
    return score

with open('datamount/character_to_prediction_index.json', "r") as f:
    char_to_num = json.load(f)

# char_to_num.items() => dict_items([(' ', 0), ('!', 1), ('#', 2), ('$', 3), ('%', 4), ('&', 5), ("'", 6), ('(', 7), (')', 8), ('*', 9), ('+', 10), (',', 11) ...

rev_character_map = {j:i for i,j in char_to_num.items()}

def decode(generated_ids):
    # generated_ids => [60 50 36 32 43 36 50 14 42 52 39 32 56 43 32 59 59 59 59 59 59 59 59 59 59 59 59 59 59 59 59 59 59]     
    
    # rev_character_map.get(id_,'')) => s
    # type(rev_character_map.get(id_,'')) => <class 'str'>
    
    # ''.join([rev_character_map.get(id_,'') for id_ in generated_ids]) =>
    # seales/kuhayla
    
    # type(''.join([rev_character_map.get(id_,'') for id_ in generated_ids])) => <class 'str'>

    return ''.join([rev_character_map.get(id_,'') for id_ in generated_ids])

dfs = []
for fold, val_fns in enumerate(val_data_fns):
    val_df = train[train['fold']==fold].copy()
    # val_df.shape => (16898, 7)
    
    val_scores = []
    for val_fn in val_fns:
        val_data = torch.load(val_fn)['generated_ids']

        val_preds = np.array([decode(generated_ids) for generated_ids in tqdm(val_data.cpu().numpy())])
        # val_preds => ['seales/kuhayla' 'https://jsi.is/hkuoka' 'dine-cin/code/' ... 'http://mirai.dinodeniya.co.jp' 'https://www.weidiadesters.com' '6001 circle']
        # len(val_preds) => 16898
        val_scores += [get_score(val_df['phrase'].values, val_preds)]
    
    # np.stack(val_scores).shape => (2, 16898)
    val_scores = np.stack(val_scores).mean(0)
    # val_scores.shape => (16898,)    

    val_df['score'] = val_scores
    dfs += [val_df]


df = pd.concat(dfs)

# df.head(3) => 
#                                path  file_id  ...  seq_len     score
# 1   train_landmarks/5414471.parquet  5414471  ...      127  0.866667
# 6   train_landmarks/5414471.parquet  5414471  ...      300  0.954545
# 12  train_landmarks/5414471.parquet  5414471  ...      114  0.923077


# train.index => RangeIndex(start=0, stop=67208, step=1)
# type(train.index) => <class 'pandas.core.indexes.range.RangeIndex'>
# getting rows order same as train DataFrame.
df = df.loc[train.index]

# df.head(3) =>
#                               path  file_id  ...  seq_len     score
# 0  train_landmarks/5414471.parquet  5414471  ...      123  0.958333
# 1  train_landmarks/5414471.parquet  5414471  ...      127  0.866667
# 2  train_landmarks/5414471.parquet  5414471  ...      236  1.000000

df['phrase_len'] = df['phrase'].str.len()

#add supplemental data
train_supp = pd.read_csv('datamount/supplemental_metadata_folded.csv')
train_supp = train_supp[train_supp['phrase_len'] < 33].copy()
train_supp['score'] = 0.5
train_supp['is_sup'] = 1

df['is_sup'] = 0
df = pd.concat([df,train_supp])

df.to_csv('datamount/train_folded_oof_supp.csv', index=False)


