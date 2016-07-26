import sys
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append("../../libsvm-3.21/python/")
from extractor import *
import numpy as np
def main():

    files = ['airplane', 'alarm-clock', 'angel', 'ant', 'apple']
    extr = Extractor('../json/')
    extr.prnt = True

    # load files
    features, isFull, classId, names = extr.loadfolders(numclass = len(files), numfull = 5, numpartial = 5,
                                                  folderList=files)

    numtestdata = 5
    testfeatures = list()
    testnames = list()
    for i in range(numtestdata):
        # take first partial sketch to be removed
        fullindex = names[0].split('_')[0] + '_' + names[0].split('_')[1]
        namesplit1 = names[0].split('_')[0]
        namesplit2 = names[0].split('_')[1]

        # get test data
        cond = [names[partialindex].split('_')[0] == namesplit1 and names[partialindex].split('_')[1] == namesplit2 for partialindex in range(len(features))]

        testfeatures.extend([features[index] for index in range(len(features)) if cond[index]])
        testnames.extend([names[index] for index in range(len(features)) if cond[index]])

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

    for index in range(len(testfeatures)):
        feature = testfeatures[index]
        name = testnames[index]

        classProb = predictor.calculateProb(feature, priorClusterProb)
        maxClass = max(classProb, key=classProb.get)
        argsort = np.argsort(classProb.values())

        highProbClass = classProb.keys[argsort[len(argsort) - 1]]
        sechighProbClass = classProb.keys[argsort[len(argsort) - 2]]

if __name__ == "__main__": main()