import sys
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append("../../libsvm-3.21/python/")
from extractor import *
import numpy as np
from featureutil import *
def main():

    files = ['airplane', 'alarm-clock', 'angel', 'ant', 'apple', 'arm', 'armchair', 'ashtray', 'axe', 'backpack', 'banana', 'barn',
             'baseball-bat', 'basket']
    extr = Extractor('../data/')
    extr.prnt = True

    # load files
    numclass = len(files)
    numfull = 10
    numpartial = 2
    numtestdata = 3
    # load files from disk
    features, isFull, classId, names = extr.loadfolders(numclass = numclass, numfull=numfull, numpartial=numpartial,
                                                        folderList=files)
    print 'Loaded ' + str(len(features)) + ' sketches'
    # partition data into test and training Picture (the last name tagged)
    features, isFull, classId, names, testfeatures, testnames, testclassid = \
        partitionfeatures(features, isFull, classId,names, numtestdata, randomPartioning = True)

    # check if any overlapping data in test and train
    for x in features:
        for y in testfeatures:
            if np.linalg.norm(x-y) < 10**-3:
                print ':('

    # train constrained k-means
    NUMPOINTS = len(features)
    constarr = getConstraints(NUMPOINTS, isFull, classId)
    ckmeans = CKMeans(constarr, np.transpose(features), k=numclass*2)
    kmeansoutput = ckmeans.getCKMeans()

    # find heterogenous clusters and train svm
    trainer = Trainer(kmeansoutput, classId, features)  ### FEATURES : TRANSPOSE?
    heteClstrFeatureId, heteClstrId = trainer.getHeterogenous()
    trainer.trainSVM(heteClstrFeatureId)
    # find the probability of given feature to belong any of athe classes
    priorClusterProb = trainer.computeProb()
    predictor = Predictor(kmeansoutput, classId)

    ncount = [0]*numclass
    for index in range(len(testfeatures)):
        feature = testfeatures[index]
        name = testnames[index]

        classProb = predictor.calculateProb(feature, priorClusterProb)
        maxClass = max(classProb, key=classProb.get)
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
    for accindex in range(min(10, len(ncount))):
        print 'N=' + str(accindex+1) + ' C=0 accuracy: ' + str(ncount[accindex])

if __name__ == "__main__": main()