import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import ipdb
from generate_ind import generate_date_2_market
from generate_ind import update_date_2_market
from code.utils import code_2_embedding_f
from code.preprocess import preprocess

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
    date_2_market0 = np.load(market0_dic_path, allow_pickle=True).item()
    date_2_market1 = np.load(market1_dic_path, allow_pickle=True).item()
    # else:
    print('ind_name is', ind_name)
    return code_2_embedding, date_2_market0, date_2_market1
    
if __name__ == '__main__':
    mode = 'train'
    query_day = '2021-01-28'
    #mode = 'update'
    if mode == 'train':
        # prepare dicts.
        print('>???>?')
        preprocess()
        generate_date_2_market('market')
        generate_date_2_market('style_market')

    elif mode == 'update':
        # 四个字典update
        keys = ['code', 'date', 'value', 'observation', 'style']
        values = [np.array(['002212.SZ']), np.array(['2021-01-28']), np.array([2.9]), np.array([np.array([1 for i in range(64)])]), np.array([np.array([1 for i in range(160)])])]
        new_data = dict(zip(keys, values))
        update_date_2_market(new_data, query_day, ind_name='market')
        new_data = dict(zip(keys, values))
        update_date_2_market(new_data, query_day, ind_name='style_market')
    else:
        print('mode unk')
        exit()
    code_2_embedding, date_2_market0, date_2_market1 = load_dicts(ind_name='market')
    s_code_2_embedding, s_date_2_market0, s_date_2_market1 = load_dicts(ind_name='style_market')
    if (query_day in date_2_market0.keys()) and (query_day in s_date_2_market0.keys()):
        ind1 = float(market_indicator(code_2_embedding, date_2_market0, '002212.SZ', query_day))
        ind2 = float(market_indicator(code_2_embedding, date_2_market1, '002212.SZ', query_day))
        ind3 = float(style_indicator(s_code_2_embedding, s_date_2_market0, '002212.SZ', query_day))
        ind4 = float(style_indicator(s_code_2_embedding, s_date_2_market1, '002212.SZ', query_day))
        print(ind1, ind2, ind3, ind4)
