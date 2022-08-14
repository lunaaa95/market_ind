import os
import torch
import torch.nn as nn
import numpy as np
import ipdb
from code.generate_ind import generate_date_2_market
from code.utils import code_2_embedding_f

def market_indicator(code_2_embedding, date_2_market, stock_code, date):
    embedding = code_2_embedding[stock_code] # array
    market = date_2_market[date] # array
    cos_sim = embedding.dot(market) / (np.linalg.norm(embedding) * np.linalg.norm(market))
    return cos_sim

def style_indicator(code_2_embedding, date_2_market, stock_code, date):
    embedding = code_2_embedding[stock_code + '-' + date] # array
    market = date_2_market[date] # array
    cos_sim = embedding.dot(market) / (np.linalg.norm(embedding) * np.linalg.norm(market))
    return cos_sim


def load_dicts(ind_name):
    if ind_name == 'market':
        code_dic_path = './data/market/code_2_embedding.npy'
        market0_dic_path = './data/market/date_2_market0.npy'
        market1_dic_path = './data/market/date_2_market1.npy'
    elif ind_name == 'style_market':
        code_dic_path = './data/market/stock_2_style.npy'
        market0_dic_path = './data/market/date_2_style_market0.npy'
        market1_dic_path = './data/market/date_2_style_market1.npy' 

    code_2_embedding = np.load(code_dic_path, allow_pickle=True).item()
    
    if os.path.exists(market0_dic_path) & os.path.exists(market1_dic_path):
        date_2_market0 = np.load(market0_dic_path, allow_pickle=True).item()
        date_2_market1 = np.load(market1_dic_path, allow_pickle=True).item()
    else:
        date_2_market0, date_2_market1 = generate_date_2_market(ind_name)
    return code_2_embedding, date_2_market0, date_2_market1
    
if __name__ == '__main__':
    # prepare dicts.
    code_2_embedding, date_2_market0, date_2_market1 = load_dicts(ind_name='market')
    s_code_2_embedding, s_date_2_market0, s_date_2_market1 = load_dicts(ind_name='style_market')
    # example
    ind1 = float(market_indicator(code_2_embedding, date_2_market0, '002212.SZ', '2021-01-28'))
    ind2 = float(market_indicator(code_2_embedding, date_2_market1, '002212.SZ', '2021-01-28'))
    ind3 = float(style_indicator(s_code_2_embedding, s_date_2_market0, '002212.SZ', '2021-01-28'))
    ind4 = float(style_indicator(s_code_2_embedding, s_date_2_market1, '002212.SZ', '2021-01-28'))
    print(ind1, ind2, ind3, ind4)


