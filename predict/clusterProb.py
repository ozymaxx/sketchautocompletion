import sys
sys.path.append("../clusterer/")
sys.path.append("../classifiers/")
import numpy as np
from ckmeans import *
from getConstraints import *
import math
from trainer import *


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



    NUMPOINTS = 9;
    NUMCLASS = 3;
    POINTSPERCLASS = NUMPOINTS/NUMCLASS

    xmin = 0;
    xmax = 100;
    ymin = 0;
    ymax = 100;

    features = np.array([np.zeros(NUMPOINTS), np.zeros(NUMPOINTS)])
    centers = np.array([np.zeros(NUMCLASS), np.zeros(NUMCLASS)])
    isFull = [np.random.randint(0, 2) for r in xrange(NUMPOINTS)]

    classId = list()
    index = 0
##################################### PREPARE FEATURES ###################################
    for i in range(0, NUMCLASS):
        classId.extend([i]*POINTSPERCLASS)
        centerx = int(np.random.random()*xmax - xmin)
        centery = int(np.random.random()*ymax - ymin)
        centers[0][i] = centerx
        centers[1][i] = centery

        for j in range(0, POINTSPERCLASS):
            datax = int(np.random.normal(loc = centerx, scale = 3))
            datay = int(np.random.normal(loc = centery, scale = 3))

            features[0][index] = datax
            features[1][index] = datay
            index += 1

    # add the remaning points, from integer division
    remainingPoints = NUMPOINTS - POINTSPERCLASS*NUMCLASS
    if (remainingPoints):
        # select the center randomly
        randc = np.random.randint(0, max(classId))
        classId.extend([randc]*remainingPoints)
        for i in range (NUMPOINTS - POINTSPERCLASS*NUMCLASS):
            datax = int(np.random.normal(loc = centers[0][randc], scale = 3))
            datay = int(np.random.normal(loc = centers[1][randc], scale = 3))
            features[0][index] = datax
            features[1][index] = datay
            index += 1

    test = getConstraints(NUMPOINTS, isFull, classId)
    k = 3
    kmeans = CKMeans(test,features,k)
    output = kmeans.getCKMeans()

    print clusterProb(np.transpose(output[1]),[1,1],computeProb(output))

if __name__ == '__main__':
    main()
    print "fin"
    #profile.run('print main(); print')


