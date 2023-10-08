import pandas
import numpy as np
#from jqmcvi import base
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics.pairwise import euclidean_distances

def InternalScores(k, content):
    Labels = content.iloc[:, -1]
    Features = content.drop(content.columns[-1], axis=1)
    Features = content.drop(content.columns[0], axis=1)

    #Davies Bouldin Index
    DBI = davies_bouldin_score(Features, Labels)

    #Silhouette Score
    SilScore =  silhouette_score(Features, Labels, metric='euclidean')

    #Calinski Harabasz Score
    CHScore = calinski_harabasz_score(Features, Labels)

    #Dunn Index
    ClusterList = []
    for i in range(0, k, 1):
        Cluster = content.loc[content.Class==i]
        if len(Cluster)!=0:
            ClusterList.append(Cluster.values)
    DI = DunnIndex(ClusterList)

    return DI, DBI, SilScore, CHScore

    
def delta(ck, cl):
    values = np.ones([len(ck), len(cl)])*10000
    
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i]-cl[j])
            
    return np.min(values)
    
def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])
    
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i]-ci[j])
            
    return np.max(values)
    
def DunnIndex(k_list):
    """ Dunn index [CVI]
    
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
    deltas = np.ones([len(k_list), len(k_list)])*1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])
        
        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas)/np.max(big_deltas)
    return di
