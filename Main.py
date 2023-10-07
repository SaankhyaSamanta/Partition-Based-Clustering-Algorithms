import pandas
import random
import numpy
import math
import argparse
import matplotlib.pyplot as Plot

from KMeansMine import *
from KMeansppMine import *
from KMediansMine import *
from KModesMine import *
from KMedoidsMine import *

from Inbuilt import *
from Internal import *
from External import *

random.seed(9)

"""
#Accuracy
correct=0
for i in range(0, datasets, 1):
        if Dataset.iloc[i]['Species']=='Iris-setosa' and content.iloc[i][-1]==1:
            correct = correct + 1
        elif Dataset.iloc[i]['Species']=='Iris-versicolor' and content.iloc[i][-1]==2:
            correct = correct + 1
        elif Dataset.iloc[i]['Species']=='Iris-virginica' and content.iloc[i][-1]==0:
            correct = correct + 1
        else:
            print('For Id %d, our algorithm predicts Class %d but actually it is %s'%(Dataset.iloc[i][0], content.iloc[i][-1], Dataset.iloc[i]['Species']))
    print("Number of correct predictions is: %d out of %d" %(correct, datasets))
"""

def Cleaning(Dataframe, datasets, attributes, HeaderRow):
    Data = pandas.DataFrame()
    for i in range(0, attributes, 1):
        List = [0]*datasets
        for j in range(0, datasets, 1):
            List[j] = Dataframe.iloc[j][i]
        #print("Original: ", List)
        if '?' not in List:
            NumList = [0]*datasets
            for j in range(0, datasets, 1):
                NumList[j] = float(List[j])
            Data[HeaderRow[i]] = NumList
            #print("To number: ", NumList)

        else:
            j=0
            while j<datasets:
                if List[j]!='?':
                    j += 1
                else:
                    k=j
                    while k<datasets and List[k]=='?':
                        k += 1
                    if k<datasets:
                        for l in range(j, k, 1):
                            List[l] = List[k]
                        j = k
                    else:
                        for l in range(j, datasets, 1):
                            List[l] = List[j-1]
                        j = datasets

            NumList = [0]*datasets
            for j in range(0, datasets, 1):
                NumList[j] = float(List[j])
            Data[HeaderRow[i]] = NumList
            #print("Number without ? :", NumList)
                
            
    return Data

def DeleteLastCol(Dataframe):
    datasets = len(Dataframe)
    attributes = len(Dataframe.columns)
    Dataframe = Dataframe.drop(Dataframe.columns[attributes-1], axis=1)
    attributes = len(Dataframe.columns)
    return Dataframe, datasets, attributes;

def Scaling(Dataframe, datasets, attributes, HeaderRow):
    for i in range(1, attributes, 1):
        Max = Dataframe.iloc[0,i]
        Min = Dataframe.iloc[0,i]
        for j in range(0, datasets, 1):
            if Max<Dataframe.iloc[j,i]:
                Max = Dataframe.iloc[j,i]
            if Min>Dataframe.iloc[j,i]:
                Min = Dataframe.iloc[j,i]
        print("For %s attribute max is %f and min is %f" %(HeaderRow[i], Max, Min))
        for j in range(0, datasets, 1):
            Dataframe.iloc[j,i] = (Max - Dataframe.iloc[j,i])/(Max - Min)
    return Dataframe

def main(args):
    print(args)

    filename = 'Csv Datasets/' + args.dataset + '.csv'
    file = open(filename, 'r')
    content = pandas.read_csv(filename)
    Dataset = pandas.read_csv(filename)
    Original = pandas.read_csv(filename)
    """
    with pandas.option_context('display.max_rows', None,'display.max_columns', None):
        print(content)

    print("\n")
    """
    datasets = 0
    attributes = 0

    content, datasets, attributes = DeleteLastCol(content)
    if args.dataset!='Iris':
        SNo = [0]*datasets
        for i in range(0, datasets, 1):
            SNo[i] = i+1
        content.insert(loc=0, column='S.No.', value = SNo)
        attributes = len(content.columns)
    List = content.columns
    if args.dataset in ['Echocardio', 'Hepatitis']:
        content = Cleaning(content, datasets, attributes, List) 
    List = content.columns
    content = Scaling(content, datasets, attributes, List)
    NumClusters = args.K

    for i in range(1, attributes, 1):
        ColName = "Scaled " + str(Dataset.columns[i])
        Col = content.iloc[:, i]
        Dataset[ColName] = Col

    if args.algorithm=='K Means':
        
        content, countM, K, SSE, MinCostM, CentroidM = KMeansMine(args.K, content, datasets, attributes)
        number = [0]*K
        if args.dataset=='Iris':
            ClassListM = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']
        elif args.dataset=='Glass':
            ClassListM = [3, 2, 4, 1, 7, 5, 6]
        elif args.dataset=='Echocardio':
            ClassListM = [1, 0, 2]
        elif args.dataset=='Heart':
            ClassListM = [1, 2]
        elif args.dataset=='Hepatitis':
            ClassListM = [2, 1]
        elif args.dataset=='Liverdisorder':
            ClassListM = [2, 1]
        
        print("K Means Mine: ")
        for i in range(0, K, 1):
            number[i] = countM[0][i]
            print("Number of data points in class %d i.e., '%s' is %d" %(i, ClassListM[i], number[i]))
        Dataset['KMeansMine'] = content['Class']
        print("Centroids: \n", CentroidM)         

        DIM, DBIM, SilScoreM, CHScoreM = InternalScores(args.K, content)
        JScoreM, FMScoreM, RIM, F1ScoreM = ExternalScores(args.K, content, Original, ClassListM)
        print("Internal Scores:")
        print("Dunn Index: ", DIM)
        print("Davies Bouldin Score: ", DBIM)
        print("Silhouette Score: ", SilScoreM)
        print("Calinski Harabasz Score: ", CHScoreM)
        print("External Scores:")
        print("Jaccard Score: ", JScoreM)
        print("Fowlkes Mallows Score: ", FMScoreM)
        print("Rand Index: ", RIM)
        print("F1 Score: ", F1ScoreM)
        
        content, datasets, attributes = DeleteLastCol(content)

        content, Centroid, count, MinCost = KMeansPy(args.K, content, False)
        number = [0]*K
        ClassList = [""]*K
        
        for i in range(0, K, 1):
            number[i] = count[0][i]
            if args.dataset=='Iris':
                if number[i]==50:
                    ClassList[i] = 'Iris-setosa'
                elif number[i]==61:
                    ClassList[i] = 'Iris-versicolor'
                else:
                    ClassList[i] = 'Iris-virginica'
            elif args.dataset=='Echocardio':
                if number[i]>70:
                    ClassList[i] = 0
                elif number[i]>30:
                    ClassList[i] = 1
                elif number[i]>10:
                    ClassList[i] = 2
        
        if args.dataset=='Glass':
            Max = []
            Dummy = []
            for l in range(0, len(number), 1):
                Dummy.append(number[l])
            i=0
            while len(Dummy)!=0:
                NMax = max(Dummy)
                #print(number, Dummy, NMax)
                for index in range(0, len(Dummy), 1):
                    if Dummy[index]==NMax:
                        break
                j=0
                while j<len(number):
                    if number[j]==NMax:
                        if i==0 and j not in Max:
                            ClassList[j] = 1
                            Max.append(j)
                        elif i==1 and j not in Max:
                            ClassList[j] = 2
                            Max.append(j)
                        elif i==2 and j not in Max:
                            ClassList[j] = 7
                            Max.append(j)
                        elif i==3 and j not in Max:
                            ClassList[j] = 3
                            Max.append(j)
                        elif i==4 and j not in Max:
                            ClassList[j] = 4
                            Max.append(j)
                        elif i==5 and j not in Max:
                            ClassList[j] = 5
                            Max.append(j)
                        elif i==6 and j not in Max:
                            ClassList[j] = 6
                            Max.append(j)
                    j = j+1
                Dummy.remove(Dummy[index])
                i = i+1
        
        if args.dataset=='Heart':
            if number[1]>number[0]:
                ClassList = [2, 1]
            else:
                ClassList = [1, 2]
        elif args.dataset=='Hepatitis':
            if number[0]>number[1]:
                ClassList = [2, 1]
            else:
                ClassList = [1, 2]
        elif args.dataset=='Liverdisorder':
            if number[0]>number[1]:
                ClassList = [2, 1]
            else:
                ClassList = [1, 2]

        print("K Means: ")
        for i in range(0, K, 1):
            print("Number of data points in class %d i.e., '%s' is %d" %(i, ClassList[i], number[i]))
        Dataset['KMeans'] = content['Class']
        print("Centroids: \n", Centroid)

        DI, DBI, SilScore, CHScore = InternalScores(args.K, content)
        JScore, FMScore, RI, F1Score = ExternalScores(args.K, content, Original, ClassList)
        print("Internal Scores:")
        print("Dunn Index: ", DI)
        print("Davies Bouldin Score: ", DBI)
        print("Silhouette Score: ", SilScore)
        print("Calinski Harabasz Score: ", CHScore)
        print("External Scores:")
        print("Jaccard Score: ", JScore)
        print("Fowlkes Mallows Score: ", FMScore)
        print("Rand Index: ", RI)
        print("F1 Score: ", F1Score)
        
        content, datasets, attributes = DeleteLastCol(content)

    elif args.algorithm=='K Meanspp':
        
        content, countM, K, SSE, MinCostM, CentroidM = KMeansppMine(args.K, content, datasets, attributes)
        number = [0]*K
        if args.dataset=='Iris':
            ClassListM = ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']
        elif args.dataset=='Glass':
            ClassListM = [1, 4, 3, 5, 2, 7, 6]
        elif args.dataset=='Echocardio':
            ClassListM = [2, 0, 1]
        elif args.dataset=='Heart':
            ClassListM = [2, 1]
        elif args.dataset=='Hepatitis':
            ClassListM = [1, 2]
        elif args.dataset=='Liverdisorder':
            ClassListM = [2, 1]
        
        print("K Meanspp Mine: ")
        for i in range(0, K, 1):
            number[i] = countM[0][i]
            print("Number of data points in class %d i.e., '%s' is %d" %(i, ClassListM[i], number[i]))
        Dataset['KMeansppMine'] = content['Class']
        print("Centroids: \n", CentroidM)

        DIM, DBIM, SilScoreM, CHScoreM = InternalScores(args.K, content)
        JScoreM, FMScoreM, RIM, F1ScoreM = ExternalScores(args.K, content, Original, ClassListM)
        print("Internal Scores:")
        print("Dunn Index: ", DIM)
        print("Davies Bouldin Score: ", DBIM)
        print("Silhouette Score: ", SilScoreM)
        print("Calinski Harabasz Score: ", CHScoreM)
        print("External Scores:")
        print("Jaccard Score: ", JScoreM)
        print("Fowlkes Mallows Score: ", FMScoreM)
        print("Rand Index: ", RIM)
        print("F1 Score: ", F1ScoreM)
        
        content, datasets, attributes = DeleteLastCol(content)

        content, Centroid, count, MinCost = KMeansPy(args.K, content, True)
        number = [0]*K
        ClassList = [""]*K
        
        for i in range(0, K, 1):
            number[i] = count[0][i]
            if args.dataset=='Iris':
                if number[i]==50:
                    ClassList[i] = 'Iris-setosa'
                elif number[i]==61:
                    ClassList[i] = 'Iris-versicolor'
                else:
                    ClassList[i] = 'Iris-virginica'
            elif args.dataset=='Echocardio':
                if number[i]>70:
                    ClassList[i] = 0
                elif number[i]>30:
                    ClassList[i] = 1
                elif number[i]>10:
                    ClassList[i] = 2

        if args.dataset=='Glass':
            Max = []
            Dummy = []
            for l in range(0, len(number), 1):
                Dummy.append(number[l])
            i=0
            while len(Dummy)!=0:
                NMax = max(Dummy)
                #print(number, Dummy, NMax)
                for index in range(0, len(Dummy), 1):
                    if Dummy[index]==NMax:
                        break
                j=0
                while j<len(number):
                    if number[j]==NMax:
                        if i==0 and j not in Max:
                            ClassList[j] = 1
                            Max.append(j)
                        elif i==1 and j not in Max:
                            ClassList[j] = 2
                            Max.append(j)
                        elif i==2 and j not in Max:
                            ClassList[j] = 7
                            Max.append(j)
                        elif i==3 and j not in Max:
                            ClassList[j] = 3
                            Max.append(j)
                        elif i==4 and j not in Max:
                            ClassList[j] = 4
                            Max.append(j)
                        elif i==5 and j not in Max:
                            ClassList[j] = 5
                            Max.append(j)
                        elif i==6 and j not in Max:
                            ClassList[j] = 6
                            Max.append(j)
                    j = j+1
                Dummy.remove(Dummy[index])
                i = i+1
        
        if args.dataset=='Heart':
            if number[1]>number[0]:
                ClassList = [2, 1]
            else:
                ClassList = [1, 2]
        elif args.dataset=='Hepatitis':
            if number[0]>number[1]:
                ClassList = [2, 1]
            else:
                ClassList = [1, 2]
        elif args.dataset=='Liverdisorder':
            if number[0]>number[1]:
                ClassList = [2, 1]
            else:
                ClassList = [1, 2]

        print("K Meanspp: ")
        for i in range(0, K, 1):
            print("Number of data points in class %d i.e., '%s' is %d" %(i, ClassList[i], number[i]))
        Dataset['KMeanspp'] = content['Class']
        print("Centroids: \n", Centroid)

        DI, DBI, SilScore, CHScore = InternalScores(args.K, content)
        JScore, FMScore, RI, F1Score = ExternalScores(args.K, content, Original, ClassList)
        print("Internal Scores:")
        print("Dunn Index: ", DI)
        print("Davies Bouldin Score: ", DBI)
        print("Silhouette Score: ", SilScore)
        print("Calinski Harabasz Score: ", CHScore)
        print("External Scores:")
        print("Jaccard Score: ", JScore)
        print("Fowlkes Mallows Score: ", FMScore)
        print("Rand Index: ", RI)
        print("F1 Score: ", F1Score)
        
        content, datasets, attributes = DeleteLastCol(content)

    elif args.algorithm=='K Medians':
        
        content, countM, K, SSE, MinCostM, CentroidM = KMediansMine(args.K, content, datasets, attributes)
        number = [0]*K
        if args.dataset=='Iris':
            ClassListM = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']
        elif args.dataset=='Glass':
            ClassListM = [6, 2, 5, 1, 7, 3, 4]
        elif args.dataset=='Echocardio':
            ClassListM = [1, 0, 2]
        elif args.dataset=='Heart':
            ClassListM = [1, 2]
        elif args.dataset=='Hepatitis':
            ClassListM = [2, 1]
        elif args.dataset=='Liverdisorder':
            ClassListM = [2, 1]
        
        print("K Medians Mine: ")
        for i in range(0, K, 1):
            number[i] = countM[0][i]
            print("Number of data points in class %d i.e., '%s' is %d" %(i, ClassListM[i], number[i]))
        Dataset['KMediansMine'] = content['Class']
        print("Centroids: \n", CentroidM)

        DIM, DBIM, SilScoreM, CHScoreM = InternalScores(args.K, content)
        JScoreM, FMScoreM, RIM, F1ScoreM = ExternalScores(args.K, content, Original, ClassListM)
        print("Internal Scores:")
        print("Dunn Index: ", DIM)
        print("Davies Bouldin Score: ", DBIM)
        print("Silhouette Score: ", SilScoreM)
        print("Calinski Harabasz Score: ", CHScoreM)
        print("External Scores:")
        print("Jaccard Score: ", JScoreM)
        print("Fowlkes Mallows Score: ", FMScoreM)
        print("Rand Index: ", RIM)
        print("F1 Score: ", F1ScoreM)
        
        content, datasets, attributes = DeleteLastCol(content)

        content, Centroid, count, MinCost, NumClusters = KMediansPy(args.K, content)
        number = [0]*K
        if args.dataset=='Iris':
            ClassList = ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']
        elif args.dataset=='Glass':
            ClassList = [1, 2, 3, 7, 0, 0, 0]
        elif args.dataset=='Echocardio':
            ClassList = [2, 1, 0]
        elif args.dataset=='Heart':
            ClassList = [2, 1]
        elif args.dataset=='Hepatitis':
            ClassList = [1, 2]
        elif args.dataset=='Liverdisorder':
            ClassList = [1, 2]
        
        print("K Medians: ")
        for i in range(0, NumClusters, 1):
            number[i] = count[0][i]
            print("Number of data points in class %d i.e., '%s' is %d" %(i, ClassList[i], number[i]))
        Dataset['KMedians'] = content['Class']
        print("Centroids: \n", Centroid)

        DI, DBI, SilScore, CHScore = InternalScores(args.K, content)
        JScore, FMScore, RI, F1Score = ExternalScores(args.K, content, Original, ClassList)
        print("Internal Scores:")
        print("Dunn Index: ", DI)
        print("Davies Bouldin Score: ", DBI)
        print("Silhouette Score: ", SilScore)
        print("Calinski Harabasz Score: ", CHScore)
        print("External Scores:")
        print("Jaccard Score: ", JScore)
        print("Fowlkes Mallows Score: ", FMScore)
        print("Rand Index: ", RI)
        print("F1 Score: ", F1Score)
        
        content, datasets, attributes = DeleteLastCol(content)

    elif args.algorithm=='K Modes':
        
        content, countM, K, SSE, MinCostM, CentroidM = KModesMine(args.K, content, datasets, attributes)
        number = [0]*K
        if args.dataset=='Iris':
            ClassListM = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']
        elif args.dataset=='Glass':
            ClassListM = [1, 2, 7, 0, 0, 0, 0]
        elif args.dataset=='Echocardio':
            ClassListM = [0, 2, 1]
        elif args.dataset=='Heart':
            ClassListM = [2, 1]
        elif args.dataset=='Hepatitis':
            ClassListM = [2, 1]
        elif args.dataset=='Liverdisorder':
            ClassListM = [1, 2]
        
        print("K Modes Mine: ")
        for i in range(0, K, 1):
            number[i] = countM[0][i]
            print("Number of data points in class %d i.e., '%s' is %d" %(i, ClassListM[i], number[i]))
        Dataset['KModesMine'] = content['Class']
        print("Centroids: \n", CentroidM)

        DIM, DBIM, SilScoreM, CHScoreM = InternalScores(args.K, content)
        JScoreM, FMScoreM, RIM, F1ScoreM = ExternalScores(args.K, content, Original, ClassListM)
        print("Internal Scores:")
        print("Dunn Index: ", DIM)
        print("Davies Bouldin Score: ", DBIM)
        print("Silhouette Score: ", SilScoreM)
        print("Calinski Harabasz Score: ", CHScoreM)
        print("External Scores:")
        print("Jaccard Score: ", JScoreM)
        print("Fowlkes Mallows Score: ", FMScoreM)
        print("Rand Index: ", RIM)
        print("F1 Score: ", F1ScoreM)

        content, datasets, attributes = DeleteLastCol(content)

        content, Centroid, count, MinCost = KModesPy(args.K, content)
        number = [0]*K
        if args.dataset=='Iris':
            ClassList = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']
        elif args.dataset=='Glass':
            ClassList = [2, 1, 3, 4, 5, 6, 7]
        elif args.dataset=='Echocardio':
            ClassList = [0, 2, 1]
        elif args.dataset=='Heart':
            ClassList = [1, 2]
        elif args.dataset=='Hepatitis':
            ClassList = [2, 1]
        elif args.dataset=='Liverdisorder':
            ClassList = [2, 1]
        
        print("K Modes: ")
        for i in range(0, K, 1):
            number[i] = count[0][i]
            print("Number of data points in class %d i.e., '%s' is %d" %(i, ClassList[i], number[i]))
        Dataset['KModes'] = content['Class']
        print("Centroids: \n", Centroid)

        DI, DBI, SilScore, CHScore = InternalScores(args.K, content)
        JScore, FMScore, RI, F1Score = ExternalScores(args.K, content, Original, ClassList)
        print("Internal Scores:")
        print("Dunn Index: ", DI)
        print("Davies Bouldin Score: ", DBI)
        print("Silhouette Score: ", SilScore)
        print("Calinski Harabasz Score: ", CHScore)
        print("External Scores:")
        print("Jaccard Score: ", JScore)
        print("Fowlkes Mallows Score: ", FMScore)
        print("Rand Index: ", RI)
        print("F1 Score: ", F1Score)

        content, datasets, attributes = DeleteLastCol(content)

    elif args.algorithm=='K Medoids':
        
        content, countM, K, MinCostM, CentroidM = KMedoidsMine(args.K, content, datasets, attributes)
        number = [0]*K
        if args.dataset=='Iris':
            ClassListM = ['Iris-versicolor', 'Iris-setosa', 'Iris-virginica']
        elif args.dataset=='Glass':
            ClassListM = [3, 1, 2, 4, 7, 6, 5]
        elif args.dataset=='Echocardio':
            ClassListM = [1, 2, 0]
        elif args.dataset=='Heart':
            ClassListM = [2, 1]
        elif args.dataset=='Hepatitis':
            ClassListM = [2, 1]
        elif args.dataset=='Liverdisorder':
            ClassListM = [2, 1]
        
        print("K Medoids Mine: ")
        for i in range(0, K, 1):
            number[i] = countM[0][i]
            print("Number of data points in class %d i.e., '%s' is %d" %(i, ClassListM[i], number[i]))
        Dataset['KMedoidsMine'] = content['Class']
        print("Centroids: \n", CentroidM)

        DIM, DBIM, SilScoreM, CHScoreM = InternalScores(args.K, content)
        JScoreM, FMScoreM, RIM, F1ScoreM = ExternalScores(args.K, content, Original, ClassListM)
        print("Internal Scores:")
        print("Dunn Index: ", DIM)
        print("Davies Bouldin Score: ", DBIM)
        print("Silhouette Score: ", SilScoreM)
        print("Calinski Harabasz Score: ", CHScoreM)
        print("External Scores:")
        print("Jaccard Score: ", JScoreM)
        print("Fowlkes Mallows Score: ", FMScoreM)
        print("Rand Index: ", RIM)
        print("F1 Score: ", F1ScoreM)

        content, datasets, attributes = DeleteLastCol(content)

        content, Centroid, count, MinCost = KMedoidsPy(args.K, content)
        number = [0]*K
        if args.dataset=='Iris':
            ClassList = ['Iris-versicolor', 'Iris-setosa', 'Iris-virginica']
        elif args.dataset=='Glass':
            ClassList = [6, 7, 3, 2, 5, 1, 4]
        elif args.dataset=='Echocardio':
            ClassList = [0, 2, 1]
        elif args.dataset=='Heart':
            ClassList = [1, 2]
        elif args.dataset=='Hepatitis':
            ClassList = [2, 1]
        elif args.dataset=='Liverdisorder':
            ClassList = [1, 2]
        
        print("K Medoids: ")
        for i in range(0, K, 1):
            number[i] = count[0][i]
            print("Number of data points in class %d i.e., '%s' is %d" %(i, ClassList[i], number[i]))
        Dataset['KMedoids'] = content['Class']
        print("Centroids: \n", Centroid)

        DI, DBI, SilScore, CHScore = InternalScores(args.K, content)
        JScore, FMScore, RI, F1Score = ExternalScores(args.K, content, Original, ClassList)
        print("Internal Scores:")
        print("Dunn Index: ", DI)
        print("Davies Bouldin Score: ", DBI)
        print("Silhouette Score: ", SilScore)
        print("Calinski Harabasz Score: ", CHScore)
        print("External Scores:")
        print("Jaccard Score: ", JScore)
        print("Fowlkes Mallows Score: ", FMScore)
        print("Rand Index: ", RI)
        print("F1 Score: ", F1Score)

        content, datasets, attributes = DeleteLastCol(content)

    EmptyDetails = [""]*len(content)
    EmptyMine = [""]*len(content)
    EmptyAlgo = [""]*len(content)
    Mine = args.algorithm + " Mine"
    Algo = args.algorithm
    
    i=0
    Min = min(NumClusters, args.K)
    while i<4*Min:
        EmptyDetails[i] = 'Class'
        EmptyMine[i] = i//4
        EmptyAlgo[i] = i//4
        i += 1
        EmptyDetails[i] = 'Label in Dataset'
        EmptyMine[i] = ClassListM[i//4]
        EmptyAlgo[i] = ClassList[i//4]
        i += 1
        EmptyDetails[i] = 'Number of elements'
        EmptyMine[i] = countM[0][i//4]
        EmptyAlgo[i] = count[0][i//4]
        i += 1
        EmptyDetails[i] = 'Centroid'
        EmptyMine[i] = CentroidM[i//4]
        EmptyAlgo[i] = Centroid[i//4]
        i += 1

    EmptyDetails[i] = 'Cost'
    EmptyMine[i] = MinCostM
    EmptyAlgo[i] = MinCost
    
    EmptyDetails[i+1] = 'Dunn Index'
    EmptyMine[i+1] = DIM
    EmptyAlgo[i+1] = DI
    
    EmptyDetails[i+2] = 'Davies Bouldin Score'
    EmptyMine[i+2] = DBIM
    EmptyAlgo[i+2] = DBI
    
    EmptyDetails[i+3] = 'Silhouette Score'
    EmptyMine[i+3] = SilScoreM
    EmptyAlgo[i+3] = SilScore
    
    EmptyDetails[i+4] = 'Calinski Harabasz Score'
    EmptyMine[i+4] = CHScoreM
    EmptyAlgo[i+4] = CHScore
    
    EmptyDetails[i+5] = 'Jaccard Score'
    EmptyMine[i+5] = JScoreM
    EmptyAlgo[i+5] = JScore
    
    EmptyDetails[i+6] = 'Fowlkes Mallows Score'
    EmptyMine[i+6] = FMScoreM
    EmptyAlgo[i+6] = FMScore
    
    EmptyDetails[i+7] = 'Rand Index'
    EmptyMine[i+7] = RIM
    EmptyAlgo[i+7] = RI
    
    EmptyDetails[i+8] = 'F1 Score'
    EmptyMine[i+8] = F1ScoreM
    EmptyAlgo[i+8] = F1Score 

    Dataset['Details'] = EmptyDetails
    Dataset[Mine] = EmptyMine
    Dataset[Algo] = EmptyAlgo
    File = args.dataset + " - " + args.algorithm + ".csv" 
    Dataset.to_csv(File, index=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', 
                        type=str, 
                        default='Iris')
    parser.add_argument('--K', 
                        type=int, 
                        default=3)
    parser.add_argument('--algorithm', 
                        type=str, 
                        default='K Means')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main(get_args())

