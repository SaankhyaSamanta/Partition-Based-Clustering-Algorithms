import pandas
import random
import numpy
import math
import matplotlib.pyplot as Plot

random.seed(9)

def KMeansMine(K, Dataframe, datasets, attributes):
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
                Dist[i][j] = Dist[i][j] + (Dataframe.iloc[i][k+1] - CentroidNew[j][k])**2
            Dist[i][j] = math.sqrt(Dist[i][j])

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
        MinDistOld = numpy.min(Dist, 1)
        CentroidOld = CentroidNew
        #print(CentroidOld)
        CentroidNew = numpy.zeros([K, attributes-1], dtype = float)
        Count = numpy.zeros([1, K], dtype = int)
        for i in range(0, datasets, 1):
            for j in range(0, K, 1):
                if Dataframe.iloc[i][attributes]==j:
                    for k in range(1, attributes, 1):
                        CentroidNew[j][k-1] = CentroidNew[j][k-1] + Dataframe.iloc[i][k]
                    Count[0][j] = Count[0][j] + 1
        
        for i in range(0, K, 1):
            for j in range(0, attributes-1, 1):
                if Count[0][i]!=0:
                    CentroidNew[i][j] = CentroidNew[i][j]/Count[0][i]
        #print(CentroidNew)

        if numpy.array_equal(CentroidOld, CentroidNew)==True:
            break
        #print(Count)
        Dataframe = Dataframe.drop(Dataframe.columns[attributes], axis=1)
        Dist = numpy.zeros([datasets, K], dtype = float)

        for i in range(0, datasets, 1):
            for j in range(0, K, 1):
                for k in range(0, attributes-1, 1):
                    Dist[i][j] = Dist[i][j] + (Dataframe.iloc[i][k+1] - CentroidNew[j][k])**2
                Dist[i][j] = math.sqrt(Dist[i][j])

        MinDistNew = numpy.min(Dist, 1)
        Class = []

        SSEOld=0
        SSENew=0
        for i in range(0, datasets, 1):
            SSEOld = MinDistOld[i]**2 + SSEOld
            SSENew = MinDistNew[i]**2 + SSENew
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
            SSEOld = MinDistOld[i]**2 + SSEOld
            SSENew = MinDistNew[i]**2 + SSENew
        SSE = SSENew - SSEOld
        #print(SSE)
        while(SSE>0.0055 or SSE<-0.0055):
            MinDistOld = numpy.min(Dist, 1)
            CentroidOld = CentroidNew
            #print(CentroidOld)
            CentroidNew = numpy.zeros([K, attributes-1], dtype = float)
            Count = numpy.zeros([1, K], dtype = int)
            for i in range(0, datasets, 1):
                for j in range(0, K, 1):
                    if Dataframe.iloc[i][attributes]==j:
                        for k in range(1, attributes, 1):
                            CentroidNew[j][k-1] = CentroidNew[j][k-1] + Dataframe.iloc[i][k]
                        Count[0][j] = Count[0][j] + 1

            for i in range(0, K, 1):
                for j in range(0, attributes-1, 1):
                    if Count[0][i]!=0:
                        CentroidNew[i][j] = CentroidNew[i][j]/Count[0][i]
            #print(CentroidNew)

            if numpy.array_equal(CentroidOld, CentroidNew)==True:
                break
            #print(Count)
            Dataframe = Dataframe.drop(Dataframe.columns[attributes], axis=1)
            Dist = numpy.zeros([datasets, K], dtype = float)

            for i in range(0, datasets, 1):
                for j in range(0, K, 1):
                    for k in range(0, attributes-1, 1):
                        Dist[i][j] = Dist[i][j] + (Dataframe.iloc[i][k+1] - CentroidNew[j][k])**2
                    Dist[i][j] = math.sqrt(Dist[i][j])

            MinDistNew = numpy.min(Dist, 1)
            Class = []
            SSEOld=0
            SSENew=0

            for i in range(0, datasets, 1):
                SSEOld = MinDistOld[i]**2 + SSEOld
                SSENew = MinDistNew[i]**2 + SSENew
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


