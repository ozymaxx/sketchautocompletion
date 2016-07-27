import sys
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append("../../libsvm-3.21/python/")
from extractor import *
import numpy as np
def main():

    files = ['airplane', 'alarm-clock', 'angel', 'ant', 'apple', 'arm', 'armchair', 'ashtray', 'axe', 'backpack', 'banana', 'barn',
             'baseball-bat', 'basket']
    extr = Extractor('../data/')
    extr.prnt = True

    # load files
    numclass = len(files)
    features, isFull, classId, names = extr.loadfolders(numclass = numclass, numfull = 60, numpartial = 5,
                                                  folderList=files)

    numtestdata = 5
    testfeatures = list()
    testnames = list()
    testclassid = list()
    for j in range(numclass):
        for i in range(numtestdata):
            # take first partial sketch to be removed

            index = 0
            while testclassid.count(classId[index]) >= numtestdata:
                index += 1

            fullindex = names[index].split('_')[0] + '_' + names[index].split('_')[1]
            namesplit1 = names[index].split('_')[0]
            namesplit2 = names[index].split('_')[1]

            # get test data mask
            cond = [names[partialindex].split('_')[0] == namesplit1 and names[partialindex].split('_')[1] == namesplit2 for partialindex in range(len(features))]

            # get only the full sketches for test data
            testfeatures.extend([features[index] for index in range(len(features)) if cond[index] and isFull[index]])
            testnames.extend([names[index] for index in range(len(features)) if cond[index] and isFull[index]])
            testclassid.extend([classId[index] for index in range(len(features)) if cond[index] and isFull[index]])

            # remove any partial sketch from the traning data
            features = [features[index] for index in range(len(features)) if not cond[index]]
            names = [names[index] for index in range(len(names)) if not cond[index]]
            isFull = [isFull[index] for index in range(len(isFull)) if not cond[index]]
            classId = [classId[index] for index in range(len(classId)) if not cond[index]]

    NUMPOINTS = len(features)
    NUMCLASS = len(files)
    constarr = getConstraints(NUMPOINTS, isFull, classId)
    ckmeans = CKMeans(constarr, np.transpose(features), k=NUMCLASS)
    kmeansoutput = ckmeans.getCKMeans()

    # find heterogenous clusters and train svm
    trainer = Trainer(kmeansoutput, classId, features)  ### FEATURES : TRANSPOSE?
    heteClstrFeatureId, heteClstrId = trainer.getHeterogenous()
    trainer.trainSVM(heteClstrFeatureId)
    # find the probability of given feature to belong any of the classes
    priorClusterProb = trainer.computeProb()
    predictor = Predictor(kmeansoutput, classId)

    n1count = 0
    n2count = 0
    for index in range(len(testfeatures)):
        feature = testfeatures[index]
        name = testnames[index]

        classProb = predictor.calculateProb(feature, priorClusterProb)
        maxClass = max(classProb, key=classProb.get)
        argsort = np.argsort(classProb.values())

        highProbClass = classProb.keys()[argsort[len(argsort) - 1]]
        sechighProbClass = classProb.keys()[argsort[len(argsort) - 2]]

        if sechighProbClass == testclassid[index]:
            n2count += 1

        if highProbClass == testclassid[index]:
            n1count += 1
            n2count += 1

    print 'N=1 accuracy: ' + str((n1count*1.0 / (numtestdata*numclass))*100)
    print 'N=2 accuracy: ' + str((n2count*1.0 / (numtestdata*numclass))*100)
if __name__ == "__main__": main()