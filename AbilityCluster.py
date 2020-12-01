from sklearn.cluster import MeanShift, estimate_bandwidth,KMeans
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import csv
from sklearn import preprocessing
from sklearn.decomposition import PCA

def load_data(path):
    return 0

def pca(data_set,n):
    pca = PCA(n_components=n,whiten=True)
    new_x = pca.fit_transform(data_set)
    return new_x

def mean_shift(data_set):
    bandwidth = estimate_bandwidth(data_set, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data_set)
    return ms.labels_

def kmean(data_set,n,rs):
    return KMeans(n_clusters=n, random_state=rs).fit_predict(data_set)
    
def get_center_ability(X,Y,index,centers):
    center=centers[index]
    distlist=[]
    center_list=[]

    for i in range(len(X)):
        if Y[i]==index:
            distlist.append(np.linalg.norm(np.array(X[i]) - np.array(center)))
            center_list.append(X[i])
    center_index=distlist.index(max(distlist))
    return center_list[center_index],center_index
