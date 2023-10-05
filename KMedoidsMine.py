import pandas
import random
from random import randint
import numpy
import math
import matplotlib.pyplot as Plot

random.seed(9)

def KMedoidsMine(K, Dataframe, datasets, attributes):
    CentroidOld = numpy.zeros([K, attributes-1], dtype = float)
    CentroidNew = numpy.zeros([K, attributes-1], dtype = float)
    DistNew = numpy.zeros([datasets, K], dtype = float)
    DistOld = numpy.zeros([datasets, K], dtype = float)
    indexnew = numpy.zeros([1, K], dtype = int)
    indexold = numpy.zeros([1, K], dtype = int)
    iteration = 0
    
    for i in range(0, K, 1):
        indexnew[0][i] = randint(0, datasets-1)
    #print(indexnew)

    for i in range(0, K, 1):
        for j in range(0, attributes-1, 1):
            CentroidNew[i][j] = Dataframe.iloc[indexnew[0][i], j+1]
    #print(CentroidNew)

    for i in range(0, datasets, 1):
        for j in range(0, K, 1):
            for k in range(0, attributes-1, 1):
                DistNew[i][j] = DistNew[i][j] + abs(Dataframe.iloc[i][k+1] - CentroidNew[j][k])

    MinDistNew = numpy.min(DistNew, 1)
    Class = [-1]*datasets

    for i in range(0, datasets, 1):
        for j in range(0, K, 1):
            if MinDistNew[i]==DistNew[i][j]:
                Class[i] = j
    Dataframe['Class'] = Class

    Count = numpy.zeros([1, K], dtype = int)
    for i in range(0, datasets, 1):
        for j in range(0, K, 1):
            if Dataframe.iloc[i][attributes]==j:
                Count[0][j] = Count[0][j] + 1

    #print(Count)
    """
    with pandas.option_context('display.max_rows', None,'display.max_columns', None):
        print(Dataframe)
    """

    count=0
    #while numpy.array_equal(CentroidOld, CentroidNew)==False and iteration<100:
    while iteration<5 and count<100:
        #print(count)
        count = count + 1
        MinDistOld = MinDistNew
        CentroidOld = CentroidNew
        DistOld = DistNew
        #print(CentroidOld)
        indexold = indexnew
        #print(indexold)
        
        replaceindex = randint(0, K-1)
        #print(replaceindex)
        indexnew[0][replaceindex] = randint(0, datasets-1)

        for i in range(0, K, 1):
            if(i!=replaceindex and indexnew[0][replaceindex]==indexnew[0][i]):
                exist=1
                break
            else:
                exist=0

        if(exist==0):
            for j in range(0, attributes-1, 1):
                CentroidNew[replaceindex][j] = Dataframe.iloc[indexnew[0][replaceindex], j+1]

        else:
            while(exist==1):
                indexnew[0][replaceindex] = randint(0, datasets-1)
                for i in range(0, K, 1):
                    if(i!=replaceindex and indexnew[0][replaceindex]==indexnew[0][i]):
                        exist=1
                        break
                    else:
                        exist=0

            for j in range(1, attributes-1, 1):
                CentroidNew[replaceindex][j] = Dataframe.iloc[indexnew[0][replaceindex], j]

        #print(CentroidNew)
        #print(indexnew)
        Dataframe = Dataframe.drop(Dataframe.columns[attributes], axis=1)
        DistNew = numpy.zeros([datasets, K], dtype = float)

        for i in range(0, datasets, 1):
            for j in range(0, K, 1):
                for k in range(0, attributes-1, 1):
                    DistNew[i][j] = DistNew[i][j] + abs(Dataframe.iloc[i][k+1] - CentroidNew[j][k])

        MinDistNew = numpy.min(DistNew, 1)
        Class = [-1]*datasets 

        CostOld=0
        CostNew=0
        for i in range(0, datasets, 1):
            CostOld = MinDistOld[i]+ CostOld
            CostNew = MinDistNew[i] + CostNew
            Cost = CostOld - CostNew

        if CostOld<=CostNew:
            CostNew = CostOld
            CentroidNew = CentroidOld
            MinDistNew = MinDistOld
            DistNew = DistOld
            iteration = iteration - 1

        #print(CentroidNew)
        #print("%d\t%d" %(CostOld, CostNew))
        
        for i in range(0, datasets, 1):
            for j in range(0, K, 1):
                if MinDistNew[i]==DistNew[i][j]:
                    Class[i] = j
                    
        Dataframe['Class'] = Class
        Count = numpy.zeros([1, K], dtype = int)
        for i in range(0, datasets, 1):
            for j in range(0, K, 1):
                if Dataframe.iloc[i][attributes]==j:
                    Count[0][j] = Count[0][j] + 1

        #print(Count)
        #print(Cost)
        
        iteration = iteration + 1
        #print(iteration)
        """
        with pandas.option_context('display.max_rows', None,'display.max_columns', None):
            print(Dataframe)
        """
            
    Count = numpy.zeros([1, K], dtype = int)
    for i in range(0, datasets, 1):
        for j in range(0, K, 1):
            if Dataframe.iloc[i][attributes]==j:
                Count[0][j] = Count[0][j] + 1
    return Dataframe, Count, K, CostNew, CentroidNew

