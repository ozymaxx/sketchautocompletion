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
        self.jsonpath = path + '/json'
        self.csvpath = path + '/csv'
        self.fio = FileIO()
        self.prnt = False

    def loadfoldercsv(self, folder):
        # if .csv file exists, then load it directly
        fullpath = self.csvpath + '/' + folder + '/' + folder + '.csv'
        if os.path.isfile(fullpath):
            print 'loading ' + str(folder) + '.csv'
            names, isFull, features = self.fio.load(fullpath)
            # because i am an idiot

            # remove .json prefix
            names = [name[0:len(name)-5] for name in names]
            isFull = [self.isFileFull(name) for name in names]
            return features, isFull, names

    def loadfolderjson(self, folder):
        fullpath = self.path + '/' + folder
        fulldir = os.listdir(fullpath)

        features = list()
        isFull = list()
        names = list()
        for file in fulldir:
            feature, isfull = self.loadfile(file)
            features.append(feature)
            isFull.append(isfull)
            names.append(file)

        return features, isFull, names

    def loadfolder(self, folder):
        fullpath = self.csvpath + '/' + folder + '/' + folder + '.csv'
        if os.path.isfile(fullpath):
            return self.loadfoldercsv(fullpath)
        else:
            return self.loadfolderjson(fullpath)

    def loadfile(self, file):
        if '.json' not in file:
            file += '.json'

        folder = file.split('_')[0]
        fullpath = self.jsonpath + '/' + folder + '/' + file

        if not os.path.isfile(fullpath):
            raise TypeError
        if self.prnt:
            print fullpath

        return featureExtract(fullpath), self.isFileFull(file)

    def isFileFull(self, filename):
        return filename.count('_') == 1

    def loadfolderscsv(self, numclass, numfull, numpartial, folderList = []):
        if folderList is []:
            folderList = os.listdir(self.csvpath)[0:numclass]

        if len(folderList) > numclass:
            folderList = folderList[0:numclass]

        features, isFull, classId,  names = list(), list(), list(), list()
        classCount = 0
        for folder in folderList:
            featuresT, isFullT, namesT = self.loadfoldercsv(folder)
            keepFull = range(1, numfull+1)
            for sketchid in keepFull:

                # too only load highest order partial sketches
                maxPartialId = max([(int(namesT[index].split('_')[2]) if not isFullT[index] else 0) for index in range(len(namesT)) if int(namesT[index].split('_')[1]) == sketchid])
                keepPartial = range(maxPartialId-numpartial+1, maxPartialId+1)
                # remove possible negative index
                keepPartial = [p for p in keepPartial if p>0]

                # cannot get class sketch id and partial id
                mask = [int(namesT[index].split('_')[1]) == sketchid and (isFullT[index] or int(namesT[index].split('_')[2]) in keepPartial)
                        for index in range(len(featuresT))]

                # apply the constraints
                features.extend([featuresT[index] for index in range(len(featuresT)) if mask[index]])
                isFull.extend([isFullT[index] for index in range(len(featuresT)) if mask[index]])
                names.extend([namesT[index] for index in range(len(featuresT)) if mask[index]])

                classId.extend([classCount] * mask.count(True))
            classCount += 1
        return features, isFull, classId, names, folderList

    def loadfoldersjson(self, numclass, numfull, numpartial, folderList = []):
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
                    fullpath = self.jsonpath + '/' + folder + '/' + sketchname + '.json'

                    try:
                        feature = self.loadfile(sketchname)
                    except:
                        continue

                    features.append(feature)
                    isFull.append(self.isFileFull(sketchname))
                    classId.append(folderList.index(folder))
                    name.append(sketchname)

        return features, isFull, classId, name

    def loadfolders(self, numclass, numfull, numpartial, folderList = []):
        return self.loadfolderscsv(numclass, numfull, numpartial, folderList)

    def processName(self, name):
        namesplt = name.split('_')
        if len(namesplt) == 2:
            return

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