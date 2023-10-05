import pandas
from sklearn.metrics import jaccard_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import rand_score
from sklearn.metrics import f1_score

def ExternalScores(k, content, Dataset, ClassList):
    Labels = content.iloc[:, -1]
    TrueLabels = [-1]*len(Dataset)
    for i in range(0, len(Dataset), 1):
        for j in range(0, k, 1):
            if Dataset.iloc[i][-1]==ClassList[j]:
                TrueLabels[i] = j

    #Jaccard Score
    JScore = jaccard_score(TrueLabels, Labels, average='micro')

    #Fowlkes Mallows Score
    FMScore = fowlkes_mallows_score(TrueLabels, Labels)

    #Rand Index
    RI = rand_score(TrueLabels, Labels)

    #F1 Score
    F1Score = f1_score(TrueLabels, Labels, average='micro')

    return JScore, FMScore, RI, F1Score
    
