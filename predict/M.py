
import sys
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append('../test/')
sys.path.append("../../libsvm-3.21/python/")
from extractor import *
import numpy as np
from featureutil import *
from FileIO import *

class M:
    """The main class to be called"""
    def __init__(self):
        self.features = list()
        self.classId = list()
        self.isFull = list()
        self.path = '../json/'
        self.trainer = None
        self.kmeansoutput = None
        self.priorClusterProb = None
        self.d = {}
        self.names = None

        self.files = ['airplane', 'alarm-clock', 'angel', 'ant', 'apple', 'arm', 'armchair', 'ashtray', 'axe', 'backpack', 'banana', 'barn',
                 'baseball-bat', 'basket']
        self.extr = Extractor('../data/')
        self.extr.prnt = True


    def getBestPredictions(self, c):
        a = sorted(c, key=c.get, reverse=True)[:5]
        l = ''
        for i in a:
            l += '&'
            l += self.files[i]
        print l
        return l


    def trainIt(self, numClass, numFull, numPartial, k):
        print "TRAININDINFAIND"
        self.features, self.isFull, self.classId, self.names, self.folderList = self.extr.loadfolders(numclass = numClass, numfull=numFull, numpartial=numPartial,
                                                                                                      folderList=self.files)
        numtestdata = 5
        print 'Loaded ' + str(len(self.features)) + ' sketches'
        #partition data into test and training

        # train constrained k-means
        NUMPOINTS = len(self.features)
        constarr = getConstraints(NUMPOINTS, self.isFull, self.classId)
        ckmeans = CKMeans(constarr, np.transpose(self.features), k=k)
        self.kmeansoutput = ckmeans.getCKMeans()
        # a = self.kmeansoutput
        # fio = FileIO()
        #
        # fio.saveAll(self.isFull,self.names,self.features,self.kmeansoutput,"lololo")
        #
        # self.names, self.isFull, self.features, self.kmeansoutput = fio.loadAll("lololo")


        # find heterogenous clusters and train svm
        self.trainer = Trainer(self.kmeansoutput, self.classId, self.features)  ### FEATURES : TRANSPOSE?
        heteClstrFeatureId, heteClstrId = self.trainer.getHeterogenous()
        self.trainer.trainSVM(heteClstrFeatureId)

    def predictIt(self, instance):
            # find the probability of given feature to belong any of athe classes
        priorClusterProb = self.trainer.computePriorProb()
        predictor = Predictor(self.kmeansoutput, self.classId)
        classProb = predictor.calculatePosteriorProb(instance, priorClusterProb)
        self.getBestPredictions(classProb)
        return classProb,max(classProb, key=classProb.get)

    def predictByPath(self,fullsketchpath):

        instance = featureExtract(fullsketchpath)
        priorClusterProb = self.trainer.computePriorProb()
        predictor = Predictor(self.kmeansoutput, self.classId)
        classProb = predictor.calculatePosteriorProb(instance, priorClusterProb)
        return classProb,max(classProb, key=classProb.get)


def main():
    # load files
    numclass = 2
    numfull = 2
    numpartial = 2
    numtestdata = 5

    m = M()
    m.trainIt(1,2,1,1)

    features=m.features
    classId = m.classId
    isFull = m.isFull
    names = m.names

    features, isFull, classId, names, testfeatures, testnames, testclassid = \
        partitionfeatures(features, isFull, classId,names, numtestdata, randomPartioning=True)

    ncount = [0]*numclass
    for index in range(len(testfeatures)):
        feature = testfeatures[index]
        name = testnames[index]
        classProb,maxClass = m.predictIt(feature)
        argsort = np.argsort(classProb.values())
        # calculate the accuracy
        ncounter = None
        for ncounter in range(numclass-1, -1, -1):
            if argsort[ncounter] == testclassid[index]:
                break
        while ncounter >= 0:
            ncount[len(ncount) - ncounter - 1] += 1
            ncounter += -1
    # change it to percentage
    print '# Class: %i \t# Full: %i \t# Partial: %i \t# Test case: %i' % (numclass, numfull, numpartial, numclass*numtestdata)
    ncount = [(x * 1.0 / (numtestdata * numclass)) * 100 for x in ncount]
    for accindex in range(min(5, len(ncount))):
        print 'N=' + str(accindex+1) + ' C=0 accuracy: ' + str(ncount[accindex])

if __name__ == "__main__": main()

