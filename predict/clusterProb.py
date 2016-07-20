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
        c_x = centers[i][0]
        c_y = centers[i][1]
        a =(instance[0],instance[1])
        b =(c_x,c_y)
        dist = computeDistance(a,b)
        probTup.append(math.exp(-1*abs(dist))*normalProb[i])
    return probTup

def calculateProb(instance, output, probability, classId):#It would increase performance to get list of instance!!!!!!!!!!!!!!!!
    #features : feature array
    #output : list [ List of Cluster nparray, List of Cluster Center nparray]
    #probability : P(Ck)
    # Returns P(Si|x)

    outDict = {}
    for i in set(classId):
        outDict[i] = 0.0

    HOC,homo = getHomogenous(output, classId)
    HEC,heto = getHeterogenous(output, classId)
    clusterPrb  = clusterProb(output[1], instance, probability)

    for i in range(len(heto)):
        modelName = "clus"+ `i` +".model"
        m = svm_load_model('../classifiers/' + modelName)
        orderedLabels = m.get_labels()
        labels, probs = svmProb(m, [instance.tolist()])
        probabilityToBeInThatCluster = clusterPrb[heto[i]]

        for c in range(len(orderedLabels)):
            probabilityToBeInThatClass = probs[0][c]
            outDict[int(orderedLabels[c])] += probabilityToBeInThatCluster * probabilityToBeInThatClass
            pass
        pass
    return outDict

def main():
########## Test Case  #######################
    files = os.listdir('../classifiers/')
    for file in files:
        extension = os.path.splitext(file)[1]
        if extension == '.model':
            os.remove('../classifiers/'+file)


    outAll = testCases.testIt(200, 12, 4)
    features = outAll[2]
    output = outAll[0]
    probability = outAll[1]
    classId = outAll[3]


    # print clusterProb(np.transpose(outAll[0][1]),[1,1],outAll[1])

    HEC, heto = getHeterogenous(output, classId)
    allSV = trainSVM(np.transpose(features), HEC, classId)
    calculateProb(np.array([0,1]), output, probability,classId)

if __name__ == '__main__':
    main()
    print "fin"
    #profile.run('print main(); print')


