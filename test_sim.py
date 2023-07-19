from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from code.utils import load_pkl
from sklearn.metrics.pairwise import cosine_similarity
import ipdb

code_2_embedding = np.load('./data/market/code_2_embedding.npy', allow_pickle=True).item()
names = np.array(list(code_2_embedding.keys()))
embeddings = np.array(list(code_2_embedding.values()))

#query_i = 1000
#query_code = names[query_i]
query_code = '002046.SZ'
query_embedding = code_2_embedding[query_code].reshape(1, -1)

print("query code is", query_code)

cos_sim = cosine_similarity(query_embedding, embeddings).squeeze() # (4724,)
sim_idx = names[np.squeeze(np.argsort(-cos_sim))[:10]]
print('top 5 similar code are:', sim_idx)

# test 下来002046和军工、风能有关，输出了前10个最相关的，看了一下概念，确实比较相关
    

'''
reduced_data = PCA(n_components=2).fit_transform(embeddings)
clustering = DBSCAN(eps=0.05, min_samples=20).fit(reduced_data)

labels = clustering.labels_
n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

unique_labels = set(labels)
print(unique_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
print(colors)

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]
    class_member_mask = (labels == k)
    xy = reduced_data[class_member_mask]

    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )
plt.savefig('./test.png')
'''
