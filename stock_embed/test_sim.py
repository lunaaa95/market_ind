from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from utils import load_pkl
from sklearn.metrics.pairwise import cosine_similarity
import ipdb

with open('data/embeddings.file', 'r') as f:
    content = f.readlines()

content = np.array([item.strip().split(' ') for item in content][1:])
names = content[:,0].astype(np.unicode_) # n,
embeddings = content[:,1:].astype(np.float32) # n, dim

query_idx = 300
print(names[query_idx])
query_embedding = embeddings[query_idx].reshape(1, -1)
cos_sim = cosine_similarity(query_embedding, embeddings).squeeze() # (4724,)
sim_idx = names[np.squeeze(np.argsort(-cos_sim))[:5]]
print(sim_idx)

    

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