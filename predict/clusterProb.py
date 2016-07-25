import sys
import os
sys.path.append('../classifiers')
sys.path.append('../clusterer')
sys.path.append('../predict')
print sys.path

import numpy as np
import math

from ckmeans import *
from getConstraints import *

import testCases
from svmutil import *

print testCases.getFeatures
from trainer import *
from svmPredict import *

def computeDistance(x,y):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sqrt(np.sum((x-y)**2))

def clusterProb(centers,instance,normalProb):
    # Returns P(Ck|x)
    probTup = []
    for i in range(len(centers)):

        dist = computeDistance(instance, centers[i])
        probTup.append(math.exp(-1*abs(dist))*normalProb[i])
    return probTup

def calculateProb(instance, kmeansoutput, priorClusterProb, classId):#It would increase performance to get list of instance!!!!!!!!!!!!!!!!
    #features : feature array
    #output : list [ List of Cluster nparray, List of Cluster Center nparray]
    #probability : P(Ck)
    # Returns P(Si|x)

    # dict of probabilities of given instance belonging to every possible class
    # initially zero
    outDict = dict.fromkeys([i for i in set(classId)], 0.0)

    homoClstrFeatureId, homoClstrId = getHomogenous(kmeansoutput, classId)
    heteClstrFeatureId, heteClstrId = getHeterogenous(kmeansoutput, classId)
    clusterPrb  = clusterProb(kmeansoutput[1], instance, priorClusterProb)
    # normalize cluster probability to add up to 1
    clusterPrb = [x/sum(clusterPrb) for x in clusterPrb]

    for clstrid in range(len(kmeansoutput[0])):
        probabilityToBeInThatCluster = clusterPrb[clstrid]
        if clstrid in homoClstrId:
            # if homogeneous cluster is empty, then do not
            # process it and continue
            if len(kmeansoutput[0][clstrid]) == 0:
                continue
            # if homogeneous then only a single class which is the first
            # feature points class
            classesInCluster = [classId[kmeansoutput[0][clstrid][0]]]
        elif clstrid in heteClstrId:
            modelName = "clus"+ ` heteClstrId.index(clstrid) ` +".model"
            m = svm_load_model('../classifiers/' + modelName)
            classesInCluster = m.get_labels()
            labels, probs = svmProb(m, [instance.tolist()])
        for c in range(len(classesInCluster)):
            probabilityToBeInThatClass = 1 if clstrid in homoClstrId else probs[0][c]
            outDict[int(classesInCluster[c])] += probabilityToBeInThatCluster * probabilityToBeInThatClass

    return outDict

def main():
########## Test Case  #######################
    # delete all ".model" files before creating them again
    files = os.listdir('../classifiers/')
    for file in files:
        extension = os.path.splitext(file)[1]
        if extension == '.model':
            os.remove('../classifiers/'+file)

    # generate data for testing
    (kmeansoutput, priorClusterProb, features, classId) = testCases.testIt(NUMPOINTS=200, NUMCLASS=12, k=30)

    # find heterogenous clusters and train svm
    heteClstrFeatureId, heteClstrId = getHeterogenous(kmeansoutput, classId)
    trainSVM(np.transpose(features), heteClstrFeatureId, classId)
    # find the probability of given feature to belong any of the classes
    outDict = calculateProb(np.array([0,1]), kmeansoutput, priorClusterProb, classId)
    print sum(outDict.values())
if __name__ == '__main__':
    main()
    print "fin"
    #profile.run('print main(); print')
