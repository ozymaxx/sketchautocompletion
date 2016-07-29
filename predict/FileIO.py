"""
FileIO
Ahmet BAGLAN
14.07.2016
"""

import numpy as np
import pandas as pd
class FileIO:
    """File class used for saving an loading """
    def __init__(self):
        self.endAll = "__all__.csv"
        self.startCent = "Centers__"
        self.startClu = "Cluster__"
    def save(self,isFull, names, feature, f):
        """"Save features to csv file
        isFull: array of 1 or 0
        names: name array of the instances
        feature: featureList
        f: name of the file to be saved
        """

        df = pd.DataFrame(data = isFull)
        df.columns = ['isFull']
        df1 = pd.DataFrame(data=feature)
        result = pd.concat([df, df1], axis=1, join_axes=[df1.index])

        df = pd.DataFrame(names)
        df.columns =['names']

        result = pd.concat([df, result], axis=1, join_axes=[result.index])

        result.to_csv(f, mode = 'w', index = False)

    def load(self, f):
        """"Load features from csv file
        INPUT
        f: name of the file to be saved
        OUTPUT
        isFull: array of 1 or 0
        names: name array of the instances
        feature: featureList
        """

        a = pd.read_csv(f)
        names = a['names'].tolist()
        isFull = a['isFull'].as_matrix()
        features = a[a.columns[2:]].as_matrix()
        return names,isFull,features


    def saveTraining(self, isFull, names, feature, kmeansoutput, f):
        """"
        Saves Training
        """

        self.save(isFull,names,feature, f)

        df = pd.DataFrame(data = kmeansoutput[0])
        df.to_csv(self.startClu + f, mode = 'w', index = False)
        df1 = pd.DataFrame(data = kmeansoutput[1])
        df1.to_csv(self.startCent + f, mode = 'w', index = False)

    def loadTraining(self, f):
        """"
        Loads Training
        """

        names,isFull,features = self.load(f)
        k1 = pd.read_csv(self.startClu+ f)
        k2 = pd.read_csv(self.startCent + f)
        k1 = list(k1.as_matrix())
        k2 = list(k2.as_matrix())
        return names,isFull,features,(k1,k2)


def main():
    fil = 'lol.csv'
    kmean = ([np.array([1,3]),np.array([2,5])],[np.array([1,2,3]),np.array([4,5,7]),np.array([6,9,19])])

    f = FileIO()
    f.saveTraining(None, None, None, kmean, fil)
    a = f.loadTraining(fil)
    # a = f.loadAll('lol.csv')

    # print a
    a = 5


if __name__ == "__main__": main()
