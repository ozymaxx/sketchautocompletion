import sys
sys.path.append('/home/semih/PycharmProjects/sketchautocompletion/classifiers')
sys.path.append('/home/semih/PycharmProjects/sketchautocompletion/clusterer')
sys.path.append('/home/semih/PycharmProjects/sketchautocompletion/predict')
print sys.path

import numpy as np
import math

from ckmeans import *
from getConstraints import *

import testCases

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
    for i in range(len(centers[0])):
        c_x = centers[0][i]
        c_y = centers[1][i]
        dist = computeDistance((instance[0],instance[1]),(c_x,c_y))
        probTup.append(math.exp(-1*abs(dist))*normalProb[i])
    return probTup

def calculateProb(features, output, probability, classId):
    #features : feature array
    #output : list [ List of Cluster nparray, List of Cluster Center nparray]
    #probability : P(Ck)
    # Returns P(Si|x)

    homo = getHomogenous(output, classId)
    heto = getHeterogenous(output, classId)
    clusterPrb  = clusterProb(output[1], features, probability)

    for i in range(len(heto)):
        modelName = "clus"+ `i` +".model"
        labels, probs = svmProb(modelName,features)
        p1 = clusterPrb[heto[i]] * probs

        pass

def main():
########## Test Case  #######################
    outAll = testCases.testIt(9, 3, 3)
    features = outAll[2]
    output = outAll[0]
    probability = outAll[1]
    classId = outAll[3]

    # print clusterProb(np.transpose(outAll[0][1]),[1,1],outAll[1])
    calculateProb(features, output, probability,classId)

if __name__ == '__main__':
    main()
    print "fin"
    #profile.run('print main(); print')


