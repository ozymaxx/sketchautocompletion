import sys
import os
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append("../../libsvm-3.21/python/")
from feature import *
from trainer import *
from FeatureExtractor import *
from shapecreator import *
from FileIO import *

class Extractor:
    def __init__(self, path):
        if (path[len(path)-1]) is '/':
            path = path[0:len(path)-1]
        self.path = path
        self.prnt = False

    def loadfolder(self, folder):
        fullpath = self.path + '/' + folder
        fulldir = os.listdir(fullpath)

        features = list()
        isFull = list()
        name = list()
        for file in fulldir:
            feature, isfull = self.loadfile(file)
            features.append(feature)
            isFull.append(isfull)
            name.append(file)

        return features, isFull, name

    def loadfile(self, file):
        if '.json' not in file:
            file += '.json'

        folder = file.split('_')[0]
        fullpath = self.path + '/' + folder + '/' + file

        if not os.path.isfile(fullpath):
            raise TypeError
        if self.prnt:
            print fullpath

        return featureExtract(fullpath), self.isFileFull(file)

    def isFileFull(self, filename):
        return filename.count('_') == 1

    def loadfolders(self, numclass, numfull, numpartial, folderList = []):
        features = list()
        isFull = list()
        classId = list()
        name = list()

        if folderList is []:
            folderList = os.listdir(self.path)[0:numclass]

        for folder in folderList:
            for sketchcounter in range(1, numfull+1):
                for partialsketchcount in range(0, numpartial+1):
                    sketchname = folder + '_' + str(sketchcounter) + ('_' + str(partialsketchcount) if partialsketchcount else '')
                    fullpath = self.path + '/' + folder + '/' + sketchname + '.json'

                    try:
                        feature = self.loadfile(sketchname)
                    except:
                        continue

                    features.append(feature)
                    isFull.append(self.isFileFull(sketchname))
                    classId.append(folderList.index(folder))
                    name.append(sketchname)

        return features, isFull, classId, name

def main():
    '''
    ext = Extractor('../json/')
    ext.prnt = True
    features, isFull, classId, name = ext.loadfolders(numclass=3, numfull=3, numpartial=2, folderList=['airplane', 'alarm-clock', 'angel', 'ant', 'apple'])


    f = FileIO()
    f.save(isFull, name, features, "noldu.csv")
    name2, full2, feature2 = f.load("noldu.csv")
    a=5

    #print ext.isFileFull('airplane_1_1')
    #print features
    '''

    pathread = '../json/'
    pathwrite = '../arrdata/'
    extr = Extractor(pathread)
    extr.prnt = True
    fio = FileIO()

    if not os.path.exists(pathwrite):
        os.makedirs(pathwrite)

    pathreaddir = os.listdir(pathread)
    for folder in pathreaddir:
        if not os.path.exists(pathwrite + folder):
            os.makedirs(pathwrite + folder)
        features, isFull, name = extr.loadfolder(folder)
        fio.save(isFull, name, features, pathwrite + folder + '/' + folder + '.csv')

    ext = Extractor('../json/')
    ext.prnt = True
    feature = ext.loadfolder('airplane')
    a=5

if __name__ == "__main__": main()