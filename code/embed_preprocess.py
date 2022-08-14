import numpy as np

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

