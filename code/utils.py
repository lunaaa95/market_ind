import pickle
import numpy as np
import ipdb

class Item():
    def __init__(self, index, code, concept, date, observation, style):
        self.index = index
        self.code = code
        self.concept = concept # array (n * 160)
        self.date = date # array (n * 1)
        self.observation = observation # (n * 31)
        self.style = style # (n * 384)


class Dataset():
    def __init__(self, code, concept, date, observation, style):
        self.code = code # array  (n * 1)
        self.concept = concept # array (n * 160)
        self.date = date # array (n * 1)
        self.observation = observation # (n * 31)
        self.style = style # (n * 384)

    def __getitem__(self, index):
        x = Item(index=index, code=self.code[index][0], concept=self.concept[index],date=self.date[index][0],
            observation=self.observation[index], style=self.style[index])
        return x
    def __len__(self):
        return len(self.code)

def save_pkl(obj, filename):
    with open('./data/mi' + filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(filename):
    with open('./data/mi' + filename, 'rb') as f:
        ret = pickle.load(f)
    return ret

def load_all(base_path):
    fields = 'code', 'concept', 'date', 'observation', 'style'
    base_path = Path(base_path)
    return {field: np.load(base_path/f'stock_{field}.npy', allow_pickle=True) for field in fields}

def code_2_embedding_f(emb_path, idx_2_code, code_2_emb_path):
    with open(emb_path, 'r') as f:
        embeddings = np.array([i.strip().split(" ") for i in f.readlines()][1:])

    idxs = [idx for idx in embeddings[:, 0].astype(int)]
    idx_embeddings = [i for i in np.array(embeddings[:,1:]).astype(np.float32)] # 4724,256
    code_2_embedding = {}
    for idx, embeddings in zip(idxs, idx_embeddings):
        if idx in idx_2_code:
            code_2_embedding[idx_2_code[idx]] = embeddings
    np.save(code_2_emb_path, code_2_embedding)
    return code_2_embedding


def walk_2_walkn():
    idx_2_code = load_pkl("idx_2_code.pkl")

    with open('./data/raw_data/concept_list.pkl', 'rb') as f:
        concept_list = pickle.load(f)
    keys = [i + len(idx_2_code) for i in range(len(concept_list))]
    idx_2_code.update(dict(zip(keys, concept_list)))

    with open('./data/stock/walks.txt', 'r') as f:
        walks = [i.strip().split(" ") for i in f.readlines()]
    walks_n = []
    for walk in walks:
        seq_len = len(walk)
        temp = []
        if int(walk[0]) < 4724:
            for i in range(0, seq_len, 2):
                temp.append(idx_2_code[int(walk[i])])
        else:
            for i in range(1, seq_len, 2):
                temp.append(idx_2_code[int(walk[i])])
        walks_n.append((" ".join(temp)) + '\n')


    with open('./data/stock/walk_n.txt', 'w') as f:
        f.writelines(walks_n)
    print('walks with names are written')
