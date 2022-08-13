import numpy as np
import pickle
from utils import save_pkl, load_pkl, Dataset, code_2_embedding_f
from pathlib import Path
from tqdm import trange, tqdm
import ipdb
from tqdm import tqdm
import pandas as pd
import os

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

dic_path = '../data/market/code_2_embedding.npy'
if os.path.exists(dic_path):
    code_2_embedding = np.load(dic_path, allow_pickle=True).item()
else:
    code_2_embedding = code_2_embedding_f()

uniq_date = [i for i in set(date.tolist())]
uniq_date.sort() # 1211个 升序

daily_concept = []
df = pd.DataFrame({'code':code, 'date':date, 'value':value})
part_n = 2

market = np.zeros((len(uniq_date), part_n, 256))

day_idx = 0
k = 300
for day in tqdm(uniq_date):
    temp_df = df[df['date'] == day]
    temp_df = temp_df.sort_values(by=['value'], ascending=False)
    market_codes = np.array([temp_df['code'][:k],
        temp_df['code'][-k:]])
    daily_concept.append(market_codes)
    temp = np.zeros((part_n, 256))
    for i in range(part_n):
        emb = np.zeros((256))
        for j in range(len(market_codes[i])):
            emb += code_2_embedding[market_codes[i,j]]
        temp[i] = emb
    market[day_idx] = temp
    day_idx += 1

np.save('../data/market/market_2side.npy', market)
exit()
    

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

