import sys
sys.path.append("../clusterer/")
sys.path.append("../classifiers/")
import numpy as np
import math

from ckmeans import *
from getConstraints import *

import testCases
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

def calculateProb(features, output, probability):
    #features : feature array
    #output : list [ List of Cluster nparray, List of Cluster Center nparray]
    #probability : P(Ck)
    # Returns P(Si|x)

    clusterPrb  = clusterProb(features, output[1], probability)

    for i in range(len(output[0])):
        modelName = "clus"+`i`+".model"
        model = svm_load_model('../classifiers/'+modelName)
        p1 = clusterPrb[i] * \
             svmProb(model, features)
        pass

def main():
########## Test Case  #######################

    outAll = testCases.testIt(9, 3, 3)
    print clusterProb(np.transpose(outAll[0][1]),[1,1],outAll[1])
    calculateProb()

if __name__ == '__main__':
    main()
    print "fin"
    #profile.run('print main(); print')


