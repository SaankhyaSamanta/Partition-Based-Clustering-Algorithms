# Partition-Based-Clustering-Algorithms

In this project I wrote my own codes for K Means, K Medians, K Modes, K Medoids and K Means++(KMeanspp) clustering algorithms in python using numpy, pandas, etc.
The performances of my algorithms and built in algorithms in sklearn/pyclustering are compared by using them on various datasets (like Iris, Glass, etc.).
The clusters formed were evaluated using internal (Calinski Harabasz Index, Davies Bouldin Index, Dunn Index, Silhouette Coefficient) and external (Rand Score, Jaccard Score, Fowlkes Mallows Score, F1 Score) clustering metrics in sklearn.metrics or direct code.

Following is description of files/directories:
1. Directory Csv datasets contains the datasets used as .csv files.
2. KMeansMine.py, KMeansppMine.py, KMediansMine.py, KModesMine.py, KMedoidsMine.py contain my implementation of the KMeans, KMeanspp, KMedians, KModes, KMedoids algorithms respectively.
3. The Inbuilt.py contains the above algorithms imported from the sklearn/pyclustering library of python.
4. Internal.py consists of the Internal Clustering metrics Dunn Index, Davies Bouldin Index, Calinski Harabasz Score, Silhouette Coefficient imported from sklearn/direct code.
5. External.py consists of the External Clustering metrics Rand Index, Jaccard Score, Fowlkes Mallows Score, F1 Score imported from sklearn.
6. Directory Results contains Results obtained by running Main.py on different datasets for various algorithms.
