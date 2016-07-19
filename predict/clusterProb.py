import sys
sys.path.append("../clusterer/")
sys.path.append("../classifiers/")
import numpy as np
from ckmeans import *
from getConstraints import *
import math
from trainer import *
import test


def computeDistance(x,y):

    x = np.asarray(x)
    y = np.asarray(y)

    return np.sqrt(np.sum((x-y)**2))

def clusterProb(centers,instance,normalProb):
    probTup = tuple()
    for i in range(len(centers[0])):
        c_x = centers[0][i]
        c_y = centers[1][i]
        dist = computeDistance((instance[0],instance[1]),(c_x,c_y))
        probTup = probTup + (math.exp(-1*abs(dist))*normalProb[i],)

    return probTup




def main():
########## Test Case  #######################

    outAll = test.testIt(9,3,3)
    print clusterProb(np.transpose(outAll[0][1]),[1,1],outAll[1])

if __name__ == '__main__':
    main()
    print "fin"
    #profile.run('print main(); print')


