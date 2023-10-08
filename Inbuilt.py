from sklearn.cluster import KMeans, kmeans_plusplus
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster import cluster_visualizer
from kmodes.kmodes import KModes
from sklearn_extra.cluster import KMedoids
import pandas
import matplotlib.pyplot as Plot
import numpy
import warnings
import random

random.seed(9)

warnings.filterwarnings('ignore')

def KMeansPy(K, content, kmeanspp):
    result = content
    result = result.drop(result.columns[0], axis=1)
    inertias=[]
    if kmeanspp==True:
        init = 'k-means++'
    else:
        init = 'random'
    for i in range(1,11):
        kmeans = KMeans(n_clusters=i, init=init)
        kmeans.fit(result.values)
        inertias.append(kmeans.inertia_)

    Plot.plot(range(1,11), inertias, marker='*', color='gold')
    Plot.title('Elbow method')
    Plot.xlabel('Number of clusters')
    Plot.ylabel('Inertia')
    Plot.show()
    
    kmeans = KMeans(n_clusters=K, init=init)
    kmeans.fit(result.values)
    PredictedClass = kmeans.predict(result.values)
    content['Class'] = PredictedClass

    datasets = len(content)
    Count = numpy.zeros([1, K], dtype = int)
    for i in range(0, datasets, 1):
        for j in range(0, K, 1):
            if content.iloc[i][-1]==j:
                Count[0][j] = Count[0][j] + 1
    
    return content, kmeans.cluster_centers_, Count, kmeans.inertia_

def KMediansPy(K, content):
    result = content
    result = result.drop(result.columns[0], axis=1)
    attributes = len(result.columns)

    """
    Inertias = []
    for l in range(1, 5):
        Initial = numpy.empty([l, attributes], dtype = float)
        for i in range(0, l, 1):
            for j in range(0, attributes, 1):
                Initial[i][j] = random.random()
        
        kmedians_instance = kmedians(result, Initial)
        kmedians_instance.process()

        medians = kmedians_instance.get_medians()
        Medians = numpy.empty([l, attributes], dtype = float)
        Clusters = kmedians_instance.get_clusters()
    
        for i in range(0, l, 1):
            for j in range(0, attributes, 1):
                Medians[i][j] = medians[i][j]

        Dist = [0]*len(content)
        Inertia=0
        for i in range(0, len(content), 1):
            for j in range(0, l, 1):
                if i in Clusters[j]:
                    for k in range(0, attributes, 1):
                        Dist[i] = Dist[i] + abs(Medians[j][k] - result.iloc[i][k])
            Inertia = Inertia + Dist[i]
        Inertias.append(Inertia)

    Plot.plot(range(1,5), Inertias, marker='*', color='gold')
    Plot.title('Elbow method')
    Plot.xlabel('Number of clusters')
    Plot.ylabel('Inertia')
    Plot.show()
    """
     
    Initial = numpy.empty([K, attributes], dtype = float)
    for i in range(0, K, 1):
        for j in range(0, attributes, 1):
            Initial[i][j] = random.random()
    kmedians_instance = kmedians(result, Initial)
    kmedians_instance.process()
    clusters = kmedians_instance.process()
    medians = kmedians_instance.get_medians()
    NumClusters = len(medians)
    Medians = numpy.empty([NumClusters, attributes], dtype = float)
    Clusters = kmedians_instance.get_clusters()
    
    for i in range(0, NumClusters, 1):
        for j in range(0, attributes, 1):
            Medians[i][j] = medians[i][j]

    Count = numpy.zeros([1, NumClusters], dtype = int)
    for j in range(0, NumClusters, 1):
        Count[0][j] = len(Clusters[j])

    List = [-1]*len(content)
    Dist = [0]*len(content)
    Inertia=0
    for i in range(0, len(content), 1):
        for j in range(0, NumClusters, 1):
            if i in Clusters[j]:
                List[i]=j
                for k in range(0, attributes, 1):
                    Dist[i] = Dist[i] + abs(Medians[j][k] - result.iloc[i][k])
                Inertia = Inertia + Dist[i]
    content['Class'] = List           

    return content, Medians, Count, Inertia, NumClusters
    

def KModesPy(K, content):
    result = content
    result = result.drop(result.columns[0], axis=1)
    cost=[]
    for i in range(1,11):
        kmodes = KModes(n_clusters=i)
        kmodes.fit(result.values)
        cost.append(kmodes.cost_)

    Plot.plot(range(1,11), cost, marker='*', color='gold')
    Plot.title('Elbow method')
    Plot.xlabel('Number of clusters')
    Plot.ylabel('Cost')
    Plot.show()

    kmodes = KModes(n_clusters=K)
    kmodes.fit(result.values)
    PredictedClass = kmodes.predict(result.values)
    content['Class'] = PredictedClass

    datasets = len(content)
    Count = numpy.zeros([1, K], dtype = int)
    for i in range(0, datasets, 1):
        for j in range(0, K, 1):
            if content.iloc[i][-1]==j:
                Count[0][j] = Count[0][j] + 1
                
    return content, kmodes.cluster_centroids_, Count, kmodes.cost_

def KMedoidsPy(K, content):
    result = content
    result = result.drop(result.columns[0], axis=1)
    inertias=[]
    for i in range(1,11):
        kmedoids = KMedoids(n_clusters=i, metric='manhattan', method='pam')
        kmedoids.fit(result.values)
        inertias.append(kmedoids.inertia_)

    Plot.plot(range(1,11), inertias, marker='*', color='gold')
    Plot.title('Elbow method')
    Plot.xlabel('Number of clusters')
    Plot.ylabel('Inertia')
    Plot.show()
    
    kmedoids = KMedoids(n_clusters=K, metric='manhattan', method='pam')
    kmedoids.fit(result.values)
    PredictedClass = kmedoids.predict(result.values)
    content['Class'] = PredictedClass

    datasets = len(content)
    Count = numpy.zeros([1, K], dtype = int)
    for i in range(0, datasets, 1):
        for j in range(0, K, 1):
            if content.iloc[i][-1]==j:
                Count[0][j] = Count[0][j] + 1
                
    return content, kmedoids.cluster_centers_, Count, kmedoids.inertia_
