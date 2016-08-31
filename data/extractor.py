import sys
import os
"""
Class for manipulating csv and json files
Can load instances with given number of class, full and partial sketches
in a super-fast manner. Can load features of 300000 sketches under a minute.

Extractor class assumes the following:
 - path + '/csv/x' includes the csv file for the class x
 - path + '/jsv/x' includes the json files for the class x

If you want to load nicicon dataset give the path of root folder of the dataset,
and use loadniciconfolders method instead. This time extractor assumes three folders;
eval, test and train, and each of these folders include 14 different nicicon clasees,
including accident and bomb.

Class Id's given on run time. and class id's are generated for each folder.
Although guarenteed to be unique -for each class-they are not necessarily the
 same for each call of extractor.
"""

from FeatureExtractor import *
from shapecreator import *
from FileIO import *

class Extractor:
    def __init__(self, path):
        if (path[len(path)-1]) is '/':
            path = path[0:len(path)-1]
        self.path = path  # main path which includes '/json' and '/csv' subfolders
        self.jsonpath = path + '/json'
        self.csvpath = path + '/csv'
        self.fio = FileIO()
        self.prnt = False
        self.debugMode = False

    def loadfoldercsv(self, folder):
        # if .csv file exists, then load it directly
        fullpath = self.csvpath + '/' + folder + '/' + folder + '.csv'
        if os.path.isfile(fullpath):
            print 'loading ' + str(folder) + '.csv'
            names, isFull, features = self.fio.load(fullpath)

            # remove .json prefix
            names = [(name[0:len(name)-5] if 'json' in name else name) for name in names]
            # because i am an idiot
            isFull = [self.isFileFull(name) for name in names]
            return features, isFull, names
        else:
            print "No such csv file as ", fullpath
            raise Exception

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
        if not folderList:
            folderList = os.listdir(self.csvpath)
            folderList.sort(key=str.lower)
            folderList = folderList[:numclass]

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

        print 'Loaded %i sketches' % len(features)
        return features, isFull, classId, names, folderList

    def loadniciconfolders(self):
        """
        loads the whole nicicon dataset
        """
        if self.path[-1] == '/':
            self.path = self.path[0:len(self.path)-1]

        whole_features, whole_isFull, whole_names, whole_classId = [], [], [], []
        symbolfolders = [self.path + "/" + f for f in os.listdir(self.path)]
        classCounter = 0
        for symbolfolder in symbolfolders:
            csvfiles = [symbolfolder + "/" + f for f in os.listdir(symbolfolder)]
            for csvfile in csvfiles:
                if os.path.isfile(csvfile):
                    print 'loading ' + str(csvfile)
                    names, isFull, features = self.fio.load(csvfile)
                    whole_features.extend(features)
                    whole_isFull.extend(isFull)
                    whole_names.extend(names)
                    whole_classId.extend([classCounter]*len(features))
                    # remove .xml postfix
                classCounter += 1
        print 'Loaded %i sketches'%len(whole_features)
        return whole_features, whole_isFull, whole_classId, whole_names

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
        """
        loads names folders  with the specified number of sketches
        :param numclass:  number of classes to be loaded.
        :param numfull: number of full sketches to be loaded for each class.
        :param numpartial:  number of partial sketches to be loaded for each full sketch.
        :param folderList: preferenced classes to be load, if empty first numclass folders are loaded
        :return:
        """
        # if no class is specified get them as the first on in the root folder
        if not folderList:
            folderList = os.listdir(self.csvpath)
            folderList.sort(key=str.lower)
            folderList = folderList[:numclass]

        if len(folderList) > numclass:
            folderList = folderList[:numclass]

        # if whole set of features for a class is requested, load it directly without shaping it
        if numfull >= 80 and numpartial >= 80:
            whole_features, whole_isFull, whole_names, whole_classId = [], [], [], []
            count = 0
            for folder in folderList:
                features, isFull, names = self.loadfoldercsv(folder)
                whole_features.extend(features)
                whole_isFull.extend(isFull)
                whole_names.extend(names)
                whole_classId.extend([count]*len(features))
                count += 1
            return whole_features, whole_isFull, whole_classId, whole_names, folderList

        features, isFull, classId, names, folderList = self.loadfolderscsv(numclass, numfull, numpartial, folderList)
        print 'Loaded ' + str(len(features)) + ' features'
        return features, isFull, classId, names, folderList

    def loadFoldersCsvParallel(self, numclass, numfull, numpartial, folderList=[]):
            if not folderList:
                folderList = os.listdir(self.csvpath)
                folderList.sort(key=str.lower)
                folderList = folderList[:numclass]

            if len(folderList) > numclass:
                folderList = folderList[0:numclass]

            features, isFull, classId, names = list(), list(), list(), list()
            classCount = 0
            for folder in folderList:
                featuresT, isFullT, namesT = self.loadfoldercsv(folder)
                #   BU DEGISECEEEEEEEKEKKKKKKKK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                features.extend(featuresT)#[:numfull*numpartial])
                isFull.extend(isFullT)#[:numfull*numpartial])
                names.extend(namesT)#[:numfull*numpartial])
                classId.extend([classCount] * len(featuresT))#[:numfull*numpartial]))
                classCount += 1
            print 'Loaded %i sketches' % len(features)
            return features, isFull, classId, names, folderList

    def loadFoldersParallel(self, numclass, numfull, numpartial, folderList = []):
        print 'in extractor'
        if not folderList:
            folderList = os.listdir(self.csvpath)
            folderList.sort(key=str.lower)
            folderList = folderList[:numclass]

        if len(folderList) > numclass:
            folderList = folderList[:numclass]

        if numfull >= 80 and numpartial >= 80:
            whole_features, whole_isFull, whole_names, whole_classId = [], [], [], []
            count = 0
            for folder in folderList:
                features, isFull, names = self.loadfoldercsv(folder)
                whole_features.extend(features)
                whole_isFull.extend(isFull)
                whole_names.extend(names)
                whole_classId.extend([count]*len(features))
                count += 1
            return whole_features, whole_isFull, whole_classId, whole_names, folderList

        features, isFull, classId, names, folderList = self.loadFoldersCsvParallel(numclass, numfull, numpartial, folderList)
        if self.debugMode:
            print names
            print 'Loaded ' + str(len(features)) + ' features'

        return features, isFull, classId, names, folderList
