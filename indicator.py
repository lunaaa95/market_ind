import os
import torch
import torch.nn as nn
import numpy as np
import ipdb
from code.generate_ind import generate_date_2_market
from code.utils import code_2_embedding_f

def indicator(code_2_embedding, date_2_market, stock_code, date):
    embedding = code_2_embedding[stock_code] # array
    market = date_2_market[date] # array
    cos_sim = embedding.dot(market) / (np.linalg.norm(embedding) * np.linalg.norm(market))
    return cos_sim

def load_dicts():
    code_dic_path = './data/market/code_2_embedding.npy'
    market0_dic_path = './data/market/date_2_market_0.npy'
    market1_dic_path = './data/market/date_2_market_1.npy'
    if os.path.exists(code_dic_path):
        code_2_embedding = np.load(code_dic_path, allow_pickle=True).item()
    else:
        code_2_embedding = code_2_embedding_f()
    
    if os.path.exists(market0_dic_path) & os.path.exists(market1_dic_path):
        date_2_market0 = np.load(market0_dic_path, allow_pickle=True).item()
        date_2_market1 = np.load(market1_dic_path, allow_pickle=True).item()
    else:
        date_2_market0, date_2_market1 = generate_date_2_market()
    return code_2_embedding, date_2_market0, date_2_market1
    
if __name__ == '__main__':
    code_2_embedding, date_2_market0, date_2_market1 = load_dicts()
    ind1 = indicator(code_2_embedding, date_2_market0, '002212.SZ', '2021-01-28')
    ind2 = indicator(code_2_embedding, date_2_market1, '002212.SZ', '2021-01-28')
    print(ind1, ind2)


