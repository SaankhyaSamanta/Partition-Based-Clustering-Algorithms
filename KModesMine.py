import pandas
import random
import numpy
import sys
from scipy import stats
import matplotlib.pyplot as Plot
import warnings

warnings.filterwarnings('ignore')

random.seed(9)

def KModesMine(K, Dataframe, datasets, attributes):
    CostOld = 0
    CostNew = 0
    CentroidOld = numpy.zeros([K, attributes-1], dtype = float)
    CentroidNew = numpy.empty([K, attributes-1], dtype = float)
    Dissimilarity = numpy.zeros([datasets, K], dtype = float)
    iteration = 0
    
    for i in range(0, K, 1):
        for j in range(0, attributes-1, 1):
            CentroidNew[i][j] = random.random()
    #print(CentroidNew)

    for i in range(0, datasets, 1):
        for j in range(0, K, 1):
            for k in range(0, attributes-1, 1):
                if Dataframe.iloc[i][k+1]!=CentroidNew[j][k]:
                    Dissimilarity[i][j] = Dissimilarity[i][j] + 1

    MinDissimilarity = numpy.min(Dissimilarity, 1)
    Class = []

    for i in range(0, datasets, 1):
        CostNew = CostNew + MinDissimilarity[i]
        valid=-1
        allottedclass=-1
        for j in range(0, K, 1):
            if MinDissimilarity[i]==Dissimilarity[i][j]:
                if j==0:
                    valid=0
                if j==1:
                    if valid==0:
                        valid = 10
                    elif valid==-1:
                        valid=1
                if j ==2:
                    if valid==-1:
                        valid=2
                    elif valid==0:
                        valid=20
                    elif valid==1:
                        valid=21
                    elif valid==10:
                        valid=210
        if valid==0:
            Class.append(0)
        elif valid==1:
            Class.append(1)
        elif valid==2:
            Class.append(2)
        elif valid==10:
            allottedclass = random.randint(0,1)
            Class.append(allottedclass)
        elif valid==21:
            allottedclass = random.randint(1,2)
            Class.append(allottedclass)
        elif valid==210:
            allottedclass = random.randint(0,2)
            Class.append(allottedclass)
        elif valid==20:
            allottedclass = random.randrange(0, 102, 2)
            allottedclass = allottedclass%4
            Class.append(allottedclass)           

    #print(CostNew)
    Dataframe['Class'] = Class

    """
    with pandas.option_context('display.max_rows', None,'display.max_columns', None):
        print(Dataframe)
    """
    
    while numpy.array_equal(CentroidOld, CentroidNew)==False and iteration<100:
        CostOld = CostNew
        CostNew = 0
        CentroidOld = CentroidNew
        #print(CentroidOld)
        CentroidNew = numpy.zeros([K, attributes-1], dtype = float)
        Count = numpy.zeros([1, K], dtype = int)
        for i in range(0, datasets, 1):
            for j in range(0, K, 1):
                if Dataframe.iloc[i][attributes]==j:
                    Count[0][j] = Count[0][j] + 1

        for i in range(0, K, 1):
            count = Count[0][i]
            List = numpy.random.uniform(low=10, high=150, size=(datasets, attributes-1))
            for j in range(0, datasets, 1):
                if Dataframe.iloc[j][attributes]==i:
                    for k in range(0, attributes-1, 1):
                        List[j][k] = Dataframe.iloc[j][k+1]
            df = pandas.DataFrame(List)
            for l in range(0, attributes-1, 1):
                CentroidNew[i][l] = df.mode().iloc[0][l]

        #print(CentroidNew)

        if numpy.array_equal(CentroidOld, CentroidNew)==True:
            CostNew = CostOld
            break
        #print(Count)
        Dataframe = Dataframe.drop(Dataframe.columns[attributes], axis=1)
        Dissimilarity = numpy.zeros([datasets, K], dtype = float)

        for i in range(0, datasets, 1):
            for j in range(0, K, 1):
                for k in range(0, attributes-1, 1):
                    if Dataframe.iloc[i][k+1]!=CentroidNew[j][k]:
                        Dissimilarity[i][j] = Dissimilarity[i][j] + 1

        MinDissimilarity = numpy.min(Dissimilarity, 1)
        Class = []

        for i in range(0, datasets, 1):
            CostNew = CostNew + MinDissimilarity[i]
            valid=-1
            allottedclass=-1
            for j in range(0, K, 1):
                if MinDissimilarity[i]==Dissimilarity[i][j]:
                    if j==0:
                        valid=0
                    if j==1:
                        if valid==0:
                            valid = 10
                        elif valid==-1:
                            valid=1
                    if j ==2:
                        if valid==-1:
                            valid=2
                        elif valid==0:
                            valid=20
                        elif valid==1:
                            valid=21
                        elif valid==10:
                            valid=210

            if valid==0:
                Class.append(0)
            elif valid==1:
                Class.append(1)
            elif valid==2:
                Class.append(2)
            elif valid==10:
                allottedclass = random.randint(0,1)
                Class.append(allottedclass)
            elif valid==21:
                allottedclass = random.randint(1,2)
                Class.append(allottedclass)
            elif valid==210:
                allottedclass = random.randint(0,2)
                Class.append(allottedclass)
            elif valid==20:
                allottedclass = random.randrange(0, 20, 2)
                allottedclass = allottedclass%4
                Class.append(allottedclass)
        #print(CostOld-CostNew)
        #print(CostNew)
        Cost = CostOld - CostNew
        Dataframe['Class'] = Class
        iteration = iteration + 1
        """
        with pandas.option_context('display.max_rows', None,'display.max_columns', None):
            print(Dataframe)
        """

    return Dataframe, Count, K, Cost, CostNew, CentroidNew

