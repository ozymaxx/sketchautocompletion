"""
FileIO
Ahmet BAGLAN
14.07.2016
"""
import numpy as np
import pandas as pd
import sys
sys.path.append('../data/')

class FileIO:
    """File class used for saving an loading """
    def __init__(self):
        self.endAll = "__all__.csv"
        self.startCent = "__Centers__"
        self.startClu = "__Cluster__"
        self.startFolder = "__folders__"

    def saveOneFeature(self, f, data):
        df = pd.DataFrame(data = data)
        df.columns = ['feat']
        df.to_csv(f, mode = 'w', index = False)

    def loadOneFeature(self,f):
        a = pd.read_csv(f)
        feature = a['feat'].as_matrix()
        return  feature


    def save(self,isFull, names, feature, f):
        """"Save features to csv file
        isFull: array of 1 or 0
        names: name array of the instances
        feature: featureList
        f: name of the file to be saved
        """
        import os
        if not os.path.exists(f[:f.rfind('/')]):
            os.mkdir(f[:f.rfind('/')])


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
        f: name of the file to be loaded
        OUTPUT
        isFull: array of 1 or 0
        names: name array of the instances
        feature: featureList
        """

        a = pd.read_csv(f)
        names = a['names'].tolist()
        isFull = a['isFull'].as_matrix()
        features = a[a.columns[2:]].as_matrix()
        return names, isFull, features


    def saveTraining(self, names, classId, isFull, feature, kmeansoutput, path, filename, saveFeatures = True):
        """"
        Saves Training
        """
        print 'Saving training data'
        import os
        if not os.path.exists(path[:path.rfind('/')]):
            os.mkdir(path[:path.rfind('/')])

        if not os.path.exists(path):
            os.mkdir(path)


        dfCl = pd.DataFrame(data= classId)
        dfCl.columns = ['ClassId']
        df = pd.DataFrame(data = isFull)
        df.columns = ['isFull']
        result = pd.concat([df, dfCl], axis=1, join_axes=[df.index])
        if(saveFeatures):
            df1 = pd.DataFrame(data=feature)
            result = pd.concat([result, df1], axis=1, join_axes=[df1.index])

        df = pd.DataFrame(names)
        df.columns =['names']
        result = pd.concat([df, result], axis=1, join_axes=[result.index])
        result.to_csv(path + '/' + filename, mode = 'w', index = False)

        df = pd.DataFrame(data = kmeansoutput[0])
        df.to_csv(path + '/' + filename + self.startClu, mode = 'w', index = False)
        df1 = pd.DataFrame(data = kmeansoutput[1])
        df1.to_csv(path + '/' + filename +self.startCent, mode = 'w', index = False)

    def loadTraining(self, f, loadFeatures = True):
        """"
        Loads Training
        """

        print 'Loading Training data ' + f
        a = pd.read_csv(f)
        names = a['names'].tolist()
        isFull = a['isFull'].as_matrix()
        classId = a['ClassId'].as_matrix()
        features = []
        if(loadFeatures):
            features = a[a.columns[3:]].as_matrix()

        k1 = pd.read_csv( f + self.startClu)
        k1 = k1.fillna("a")
        k1 = list(k1.as_matrix())
        for i in range(len(k1)):
            a = list(k1[i])
            a = filter(lambda a: a != "a", a)
            k1[i] = np.array(a)

        k2 = pd.read_csv( f+self.startCent)
        k2 = list(k2.as_matrix())

        folderList = []
        for name in names:
            className = name.split('_')[0]
            if className not in folderList:
                folderList.append(className)

        folderList.sort()
        return names, classId, isFull, features, (k1, k2), folderList

        # return (k1,k2)
def main():
    pass
    fil = 'lol.csv'
    kmean = ([np.array([1,3]),np.array([2,5,8])],[np.array([1,2,3]),np.array([4,5,7])])

    f = FileIO()
    f.saveTraining(None, None, None, None,kmean, fil)
    a = f.loadTraining(fil)

    # # print a
    a = 5

if __name__ == "__main__": main()
