import numpy as np
import pickle
from utils import save_pkl, load_pkl, Dataset, code_2_embedding_f
from pathlib import Path
from tqdm import trange, tqdm
import ipdb
from tqdm import tqdm
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gene_market', type=bool, default=False, help="If True, regenerate market2side.npy to data/market")
parser.add_argument('--gene_style', type=bool, default=False, help="If True, regenerate style2side.npy to data/market")
args = parser.parse_args()


def load_all(base_path):
    fields = 'code', 'concept', 'date', 'observation', 'style'
    base_path = Path(base_path)
    return {field: np.load(base_path/f'stock_{field}.npy', allow_pickle=True) for field in fields}

all_data = load_all('../data/raw_data')

dataset = Dataset(**all_data)
code = all_data['code'].squeeze()
concept = all_data['concept'].squeeze()
date = all_data['date'].squeeze()
observation = all_data['observation'] # n, 31
value = observation[:, 1].squeeze() # n,

with open('../data/raw_data/style_list.pkl', 'rb') as f:
    style_list = pickle.load(f)

key_words = ['PEG', 'PETTM', '市盈率','市值']
temp_dic = dict(zip(style_list, range(len(style_list))))
choose_idx = []
choose_feats = []
for k, v in temp_dic.items():
    for j in key_words:
        if j in k:
            choose_idx.append(v)
            choose_feats.append(k)
        pass
trans_vec = np.array([0.1, 0.3, 0.5, 0.7, 0.9, # 全行业PEG
    0.1, 0.3, 0.5, 0.7, 0.9, # 全行业PETTM
    0.175, 0.5, 0.7, 0.85, 0.975, # 市值
    0.1, 0.3, 0.5, 0.7, 0.9, # 行业内PEG
    0.1, 0.3, 0.5, 0.7, 0.9, # 行业内PETTM
    0.1, 0.3, 0.5, 0.7, 0.9]) # 行业内动态市盈率
temp = all_data['style'][:, choose_idx] * trans_vec # n, 30
style = np.zeros((temp.shape[0], 6))
for i in range(6):
    style[:, i] = np.sum(temp[:, i * 5: (i + 1) * 5], axis=1)

uniq_date = [i for i in set(date.tolist())]
uniq_date.sort() # 1211个 升序

part_n = 2

df = pd.DataFrame({'code':code, 'date':date, 'value':value})

# 生成 market 数据
## 获取 code_2_embedding字典
dic_path1 = '../data/market/code_2_embedding.npy'
if os.path.exists(dic_path1):
    code_2_embedding = np.load(dic_path1, allow_pickle=True).item()
else:
    code_2_embedding = code_2_embedding_f()



## 生成 market 数据
if (not args.gene_market) and os.path.exists('../data/market/market_2side.npy'):
    pass
else:
    print("--------重新生成 market_2side.npy---------")
    market = np.zeros((len(uniq_date), part_n, 256))
    day_idx = 0
    k = 300
    for day in tqdm(uniq_date):
        temp_df = df[df['date'] == day]
        temp_df = temp_df.sort_values(by=['value'], ascending=False)
        market_codes = np.array([temp_df['code'][:k],
            temp_df['code'][-k:]])
        temp = np.zeros((part_n, 256))
        for i in range(part_n):
            emb = np.zeros((256))
            for j in range(len(market_codes[i])):
                emb += code_2_embedding[market_codes[i,j]]
            temp[i] = emb
        market[day_idx] = temp
        day_idx += 1

    np.save('../data/market/market_2side.npy', market)
    print("saved")


#  生成style 数据
if (not args.gene_style) and os.path.exists('../data/market/stock_2_style.npy'):
    pass
else:
    idx_2_style = dict(zip(range(style.shape[0]), style))
    stock_style_dict = {}
    print('--------重新生成 stock_2_style.npy ---------')
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        k = row['code'] + '-' + row['date']
        stock_style_dict[k] = idx_2_style[index]
    np.save('../data/market/stock_2_style.npy', stock_style_dict) 
    print('saved')

    df['idx'] = range(style.shape[0])
    print("--------重新生成 style_2side.npy ---------")
    style_market = np.zeros((len(uniq_date), part_n, 6))
    day_idx = 0
    k = 300
    for day in tqdm(uniq_date):
        temp_df = df[df['date'] == day]
        temp_df = temp_df.sort_values(by=['value'], ascending=False)
        idxs = np.array([temp_df['idx'][:k],
            temp_df['idx'][-k:]])
        temp = np.zeros((part_n, 6))
        for i in range(part_n):
            emb = np.zeros((6))
            for j in range(len(idxs[i])):
                emb += idx_2_style[idxs[i,j]]
            temp[i] = emb
        style_market[day_idx] = temp
        day_idx += 1

    np.save('../data/market/style_2side.npy', style_market)
    print("saved")
exit()



    

code_names = list(set((code[:,0].tolist()))) # 4724
code_names.sort()
date_names = list(set((date[:,0].tolist()))) # 1211
date_names.sort()

idx_2_code = dict((k,v) for k,v in enumerate(code_names))
code_2_idx = dict((k,v) for v,k in idx_2_code.items())

idx_2_date = dict((k,v) for k,v in enumerate(date_names))
date_2_idx = dict((k,v) for v,k in idx_2_date.items())

dy_concepts = np.zeros((len(code_names), len(date_names), concept.shape[1]))
dominator = np.zeros(len(code_names))
for item in tqdm(dataset):
    idx_of_code = code_2_idx[item.code]
    idx_of_date = date_2_idx[item.date]
    dy_concepts[idx_of_code, idx_of_date] = item.concept
    dominator[idx_of_code] += 1

save_pkl(idx_2_code, 'idx_2_code.pkl')
save_pkl(code_2_idx, 'code_2_idx.pkl')
save_pkl(idx_2_date, 'idx_2_date.pkl')
save_pkl(date_2_idx, 'date_2_idx.pkl')

mean_concepts = np.sum(dy_concepts, axis=1)/dominator.reshape(-1,1) # len(code_names) * concept.shape[1]
rows, cols = mean_concepts.shape
adjlist = []
for i in range(rows):
    for j in range(cols):
        if mean_concepts[i][j] < 1e-5:
            continue
        j_no = j + rows
        adjlist.append("{} {} {}\n".format(i, j_no, mean_concepts[i][j]))

with open('data/adjlist.txt', 'w') as f:
    f.writelines(adjlist)

print('adjlist write sucess!')


'''
concept_dict = {}
for i in trange(len(code)):
    code_i = code[i][0]
    concept_i = concept[i].reshape(1, -1)
    if code_i in concept_dict:
        concept_dict[code_i] = np.vstack((concept_dict[code_i], concept_i))
    else:
        concept_dict[code_i] = concept_i

concepts = {}
for k, v in concept_dict.items():
    v = np.mean(v, axis=0)
    concepts[k] = v

save_pkl(concepts, 'concepts.pkl')
'''
# concepts = load_pkl('concepts.pkl')

