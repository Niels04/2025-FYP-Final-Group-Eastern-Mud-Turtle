import numpy as np
from sklearn.cluster import KMeans

def color_dominance(mask, clusters = 5):
    print(mask)
    flat_im = np.reshape(mask, (-1, 3)) 
    k_means = KMeans(n_clusters=clusters, n_init=10, random_state=0)
    k_means.fit(flat_im)
    return len(k_means.cluster_centers_)
