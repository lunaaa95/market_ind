import numpy as np
import pickle
from code.utils import save_pkl, load_pkl, Dataset, code_2_embedding_f
from pathlib import Path
from tqdm import trange, tqdm
import ipdb
from tqdm import tqdm
import pandas as pd
import os
import argparse
from code.generate_adjlist import generate_adjlist
from code.deepwalk.run import process

parser = argparse.ArgumentParser()
parser.add_argument('--gene_market', type=bool, default=False, help="If True, regenerate market2side.npy to data/market")
parser.add_argument('--gene_style', type=bool, default=False, help="If True, regenerate style2side.npy to data/market")
parser.add_argument('--gene_concept', type=bool, default=False, help="If True, regenerate stock_2_concept.npy to data/market")
args = parser.parse_args()

# CONST
part_n = 2

def load_all(base_path):
    fields = 'code', 'concept', 'date', 'observation', 'style'
    base_path = Path(base_path)
    return {field: np.load(base_path/f'stock_{field}.npy', allow_pickle=True) for field in fields}

def load_style(style_path, all_data):
    with open(style_path, 'rb') as f:
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
    return style

def update_market_file(new_data, date, market_file_path, part_n=2, k=300):
    # new_data:df:3col: 'date', 'value', 'code'
    new_data.pop('style')
    new_data.pop('observation')
    new_data = pd.DataFrame(new_data)
    code_2_embedding = np.load('./data/market/code_2_embedding.npy', allow_pickle=True).item()
    temp_df = new_data[new_data['date'] == date]
    temp_df = temp_df.sort_values(by=['value'], ascending=False)
    market_codes = np.array([temp_df['code'][:k], temp_df['code'][-k:]])
    temp = np.zeros((part_n, 256))
    for i in range(part_n):
        emb = np.zeros((256))
        for j in range(len(market_codes[i])):
            emb += code_2_embedding[market_codes[i,j]]
        temp[i] = emb 
    market = np.load(market_file_path)
    market = np.r_[market, np.expand_dims(temp, axis=0)]
    np.save(market_file_path, market)
    print('-------------', market_file_path, 'update!-------')

def update_style_file(new_data, date, style_file_path, part_n=2, k=300):
# new_data:df: all_data 形式的输入
# style_fule_path: './data/market/stock_2_side.npy'
    
    stock_2_style = np.load('./data/market/stock_2_style.npy', allow_pickle=True).item() # (4274234个key，每个key对应(6,) array
    style = load_style('./data/raw_data/style_list.pkl', new_data)
    new_data.pop('style')
    new_data.pop('observation')
    new_data['idx'] = [i for i in range(style.shape[0])]
    new_data = pd.DataFrame(new_data)
    idx_2_style = dict(zip(range(style.shape[0]), style))
    for index, row in new_data.iterrows():
        k = row['code'] + '-' + row['date']
        stock_2_style[k] = idx_2_style[index]
    np.save('./data/market/stock_2_style.npy', stock_2_style) 

    style_2side = np.load(style_file_path) # (1211,2,6)
    day_idx = len(style_2side)
    temp_df = new_data[new_data['date'] == date]
    temp_df = temp_df.sort_values(by=['value'], ascending=False)
    idxs = np.array([temp_df['idx'][:300], temp_df['idx'][-300:]])
    temp = np.zeros((part_n, 6))
    for i in range(part_n):
        emb = np.zeros((6))
        for j in range(len(idxs[i])):
            emb += idx_2_style[idxs[i,j]]
        temp[i] = emb
    style_2side = np.r_[style_2side, np.expand_dims(temp, axis=0)]
    np.save('./data/market/style_2side.npy', style_2side)

def preprocess():
    base_path = './data'
    raw_data_path = base_path + '/raw_data'
    indicator_path = base_path + '/market'

    all_data = load_all(raw_data_path)
    dataset = Dataset(**all_data)
    code = all_data['code'].squeeze()
    concept = all_data['concept'].squeeze()
    date = all_data['date'].squeeze()
    observation = all_data['observation'] # n, 31
    value = observation[:, 1].squeeze() # n,

    uniq_date = [i for i in set(date.tolist())]
    uniq_date.sort() # 1211个 升序

    style = load_style(raw_data_path + '/style_list.pkl', all_data)

    df = pd.DataFrame({'code':code, 'date':date, 'value':value})
    
    # 生成 个股本质embeddings
    ## 准备deepwalk需要的adjlist
    adjlist, idx_2_code = generate_adjlist(all_data, base_path + '/adjlist.txt')

    ## 进行deepwalk
    class Deepwalk_paras():
        def __init__(self, input_path, output_path, max_memory_data_size= 1e9, forma= 'adjlist', is_undirected=True, vertex_freq_degree=False, number_walks=10, representation_size=256, seed=0, walk_length=40, window_size=5, workers=1):
            self.max_memory_data_size=max_memory_data_size
            self.format = forma
            self.undirected = is_undirected
            self.number_walks=number_walks
            self.vertex_freq_degree= vertex_freq_degree
            self.representation_size=representation_size
            self.seed=seed
            self.walk_length=walk_length
            self.window_size= window_size
            self.workers = workers
            self.input = input_path
            self.output = output_path

    paras = Deepwalk_paras(input_path=base_path+'/adjlist.txt', output_path=base_path + '/embeddings.txt')
    process(paras)
    print('---------------deepwalk complete----------------')
    
    # 生成 market 数据
    ## 获取 code_2_embedding字典
    dic_path1 = indicator_path + '/code_2_embedding.npy'
    if os.path.exists(dic_path1):
        code_2_embedding = np.load(dic_path1, allow_pickle=True).item()
    else:
        ## 此处缺少个embedding文件没有跑通
        print('generating file:', indicator_path + '/code_2_embedding.npy')
        code_2_embedding = code_2_embedding_f(base_path + '/embeddings.txt', idx_2_code, indicator_path + '/code_2_embedding.npy')

    ## 生成 market 数据
    if (not args.gene_market) and os.path.exists(indicator_path + '/market_2side.npy'):
        print('market_2side.npy already exist.')
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

        np.save(indicator_path + '/market_2side.npy', market)
        print("saved")


    #  生成style 数据
    if (not args.gene_style) and os.path.exists(indicator_path + '/stock_2_style.npy'):
        print('stock_2_style.npy already exist.')
        pass
    else:
        idx_2_style = dict(zip(range(style.shape[0]), style))
        stock_style_dict = {}
        print('--------重新生成 stock_2_style.npy ---------')
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            k = row['code'] + '-' + row['date']
            stock_style_dict[k] = idx_2_style[index]
        np.save(indicator_path + '/stock_2_style.npy', stock_style_dict) 
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

        np.save(indicator_path + '/style_2side.npy', style_market)
        print("saved")

    #  生成concept 数据
    if (not args.gene_concept) and os.path.exists(indicator_path + 'stock_2_concept.npy'):
        print('stock_2_concept.npy already exist.')
        pass
    else:
        idx_2_concept = dict(zip(range(concept.shape[0]), concept))
        stock_concept_dict = {}
        print('--------重新生成 stock_2_concept.npy ---------')
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            k = row['code'] + '-' + row['date']
            stock_concept_dict[k] = idx_2_concept[index]
        np.save(indicator_path + 'stock_2_concept.npy', stock_concept_dict) 
        print('saved')
if __name__ == '__main__':
    preprocess()


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

