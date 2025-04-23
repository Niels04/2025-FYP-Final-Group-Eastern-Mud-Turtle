import numpy as np
from sklearn.cluster import KMeans

def fCHEESE_extractor(mask, clusters = 15) -> int:
    """Function to extract the number of clusters in the mask.
    
    :param mask: The mask.
    :param clusters: The clusters.
    
    :return: Number of clusters.
    
    """

    # reshape the image to extract the RGB values
    flat_im = np.reshape(mask, (-1, 3))

    # fit KMeans to the reshaped image to extract n. of clusters 
    k_means = KMeans(n_clusters=clusters, n_init=10, random_state=0)
    k_means.fit(flat_im)
    
    return len(k_means.cluster_centers_)
