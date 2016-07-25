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
        for file in fulldir:
            feature, isfull = self.loadfile(file, True)
            features.append(feature)
            isFull.append(isfull)
        return features, isFull

    def loadfile(self, file):
        if '.json' not in file:
            file += '.json'

        folder = file.split('_')[0]
        fullpath = self.path + '/' + folder + '/' + file
        if self.prnt:
            print fullpath
        if not os.path.isfile(fullpath):
            raise TypeError
        return featureExtract(fullpath), self.isFileFull(file)

    def isFileFull(self, filename):
        return filename.count('_') > 1

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
    ext = Extractor('../json/')
    ext.prnt = True
    features, isFull, classId, name = ext.loadfolders(numclass=5, numfull=10, numpartial=2, folderList=['airplane', 'alarm-clock', 'angel', 'ant', 'apple'])

    print ext.isFileFull('airplane_1_1')
    print features

if __name__ == "__main__": main()