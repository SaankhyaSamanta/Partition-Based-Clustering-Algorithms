import pandas
import random
import numpy
import sys
import matplotlib.pyplot as Plot

random.seed(9)

def KMediansMine(K, Dataframe, datasets, attributes):
    CentroidOld = numpy.zeros([K, attributes-1], dtype = float)
    CentroidNew = numpy.empty([K, attributes-1], dtype = float)
    Dist = numpy.zeros([datasets, K], dtype = float)
    iteration = 0
    SSEOld=0
    SSENew=0
    SSE=0
    
    for i in range(0, K, 1):
        for j in range(0, attributes-1, 1):
            CentroidNew[i][j] = random.random()
    #print(CentroidNew)

    for i in range(0, datasets, 1):
        for j in range(0, K, 1):
            for k in range(0, attributes-1, 1):
                Dist[i][j] = Dist[i][j] + abs(Dataframe.iloc[i][k+1] - CentroidNew[j][k])

    MinDistNew = numpy.min(Dist, 1)
    Class = []

    for i in range(0, datasets, 1):
        for j in range(0, K, 1):
            if MinDistNew[i]==Dist[i][j]:
                Class.append(j)
    Dataframe['Class'] = Class

    """
    with pandas.option_context('display.max_rows', None,'display.max_columns', None):
        print(Dataframe)
    """
    
    while numpy.array_equal(CentroidOld, CentroidNew)==False and iteration<100:
        Classes = numpy.ones([K, attributes-1, datasets], dtype=float)
        MinDistOld = numpy.min(Dist, 1)
        CentroidOld = CentroidNew
        #print(CentroidOld)
        CentroidNew = numpy.zeros([K, attributes-1], dtype = float)
        Count = numpy.zeros([1, K], dtype = int)
        for i in range(0, datasets, 1):
            for j in range(0, K, 1):
                if Dataframe.iloc[i][attributes]==j:
                    for k in range(0, attributes-1, 1):
                        Classes[j][k][i] = Dataframe.iloc[i][k+1]
                    Count[0][j] = Count[0][j] + 1

        ClassesSorted = numpy.sort(Classes)
        
        for i in range(0, K, 1):
            for j in range(0, attributes-1, 1):
                if Count[0][i]!=0:
                    if Count[0][i]%2==1:
                        count = Count[0][i]
                        CentroidNew[i][j] = ClassesSorted[i][j][count//2]
                    else:
                        count = Count[0][i]
                        CentroidNew[i][j] = 0.5*(ClassesSorted[i][j][count//2 - 1] + ClassesSorted[i][j][count//2])
                               
        #print(CentroidNew)

        if numpy.array_equal(CentroidOld, CentroidNew)==True:
            break
        #print(Count)
        Dataframe = Dataframe.drop(Dataframe.columns[attributes], axis=1)
        Dist = numpy.zeros([datasets, K], dtype = float)

        for i in range(0, datasets, 1):
            for j in range(0, K, 1):
                for k in range(0, attributes-1, 1):
                    Dist[i][j] = Dist[i][j] + abs(Dataframe.iloc[i][k+1] - CentroidNew[j][k])

        MinDistNew = numpy.min(Dist, 1)
        Class = []

        SSEOld=0
        SSENew=0
        for i in range(0, datasets, 1):
            SSEOld = MinDistOld[i] + SSEOld
            SSENew = MinDistNew[i] + SSENew
        SSE = SSEOld - SSENew
        #print(SSE)

        for i in range(0, datasets, 1):
            for j in range(0, K, 1):
                if MinDistNew[i]==Dist[i][j]:
                    Class.append(j)
        Dataframe['Class'] = Class
        iteration = iteration + 1
        """
        with pandas.option_context('display.max_rows', None,'display.max_columns', None):
            print(Dataframe)
        """

    if iteration==100 and numpy.array_equal(CentroidOld, CentroidNew)==False:
        SSEOld=0
        SSENew=0
        for i in range(0, datasets, 1):
            SSEOld = MinDistOld[i] + SSEOld
            SSENew = MinDistNew[i] + SSENew
        SSE = SSENew - SSEOld
        #print(SSE)
        while(SSE>0.16 or SSE<-0.16):
            MinDistOld = numpy.min(Dist, 1)
            CentroidOld = CentroidNew
            #print(CentroidOld)
            CentroidNew = numpy.zeros([K, attributes-1], dtype = float)
            Classes = numpy.ones([K, attributes-1, datasets], dtype=float)
            Count = numpy.zeros([1, K], dtype = int)

            for i in range(0, datasets, 1):
                for j in range(0, K, 1):
                    if Dataframe.iloc[i][attributes]==j:
                        for k in range(0, attributes-1, 1):
                            Classes[j][k][i] = Dataframe.iloc[i][k+1]
                        Count[0][j] = Count[0][j] + 1

            ClassesSorted = numpy.sort(Classes)

            for i in range(0, K, 1):
                for j in range(0, attributes-1, 1):
                    if Count[0][i]!=0:
                        if Count[0][i]%2==1:
                            count = Count[0][i]
                            CentroidNew[i][j] = ClassesSorted[i][j][count//2]
                        else:
                            count = Count[0][i]
                            CentroidNew[i][j] = 0.5*(ClassesSorted[i][j][count//2 - 1] + ClassesSorted[i][j][count//2])
                               
            #print(CentroidNew)

            if numpy.array_equal(CentroidOld, CentroidNew)==True:
                break
            #print(Count)
            Dataframe = Dataframe.drop(Dataframe.columns[attributes], axis=1)
            Dist = numpy.zeros([datasets, K], dtype = float)

            for i in range(0, datasets, 1):
                for j in range(0, K, 1):
                    for k in range(0, attributes-1, 1):
                        Dist[i][j] = Dist[i][j] + abs(Dataframe.iloc[i][k+1] - CentroidNew[j][k])

            MinDistNew = numpy.min(Dist, 1)
            Class = []
            SSEOld=0
            SSENew=0

            for i in range(0, datasets, 1):
                SSEOld = MinDistOld[i] + SSEOld
                SSENew = MinDistNew[i] + SSENew
            SSE = SSEOld - SSENew
            #print(SSE)

            for i in range(0, datasets, 1):
                for j in range(0, K, 1):
                    if MinDistNew[i]==Dist[i][j]:
                        Class.append(j)
            Dataframe['Class'] = Class
            """
            with pandas.option_context('display.max_rows', None,'display.max_columns', None):
                print(Dataframe)
            """


    Count = numpy.zeros([1, K], dtype = int)
    for i in range(0, datasets, 1):
        for j in range(0, K, 1):
            if Dataframe.iloc[i][attributes]==j:
                Count[0][j] = Count[0][j] + 1
    return Dataframe, Count, K, SSE, SSENew, CentroidNew


