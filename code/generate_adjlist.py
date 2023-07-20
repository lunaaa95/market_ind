import numpy as np
import ipdb
from tqdm import tqdm
from code.utils import save_pkl
from code.utils import Dataset


def generate_adjlist(all_data, save_path):
    dataset = Dataset(**all_data)
    code = all_data['code'].squeeze()
    concept = all_data['concept'].squeeze()
    date = all_data['date'].squeeze()
    observation = all_data['observation'] # n, 31
    value = observation[:, 1].squeeze() # n,
    code_names = list(set((code.tolist()))) # 4724
    code_names.sort()
    date_names = list(set((date.tolist()))) # 1211
    date_names.sort()

    idx_2_code = dict((k,v) for k,v in enumerate(code_names))
    code_2_idx = dict((k,v) for v,k in idx_2_code.items())

    idx_2_date = dict((k,v) for k,v in enumerate(date_names))
    date_2_idx = dict((k,v) for v,k in idx_2_date.items())

    dy_concepts = np.zeros((len(code_names), len(date_names), concept.shape[1]), dtype='f')
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

    with open(save_path, 'w') as f:
        f.writelines(adjlist)

    print('adjlist write sucess!')
    return adjlist, idx_2_code

