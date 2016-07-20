import numpy as np
import sys
sys.path.append("../clusterer/")
sys.path.append("../classifiers/")
from ckmeans import *
from getConstraints import *
from trainer import *


def testIt(NUMPOINTS, NUMCLASS, k):  #k is already the number of clusters

    (features,isFull,classId)  = getFeatures(NUMPOINTS,NUMCLASS,k)
    test = getConstraints(NUMPOINTS, isFull, classId)
    kmeans = CKMeans(test,features,k)
    output = kmeans.getCKMeans()
    probabilities = computeProb(output)

    return (output,probabilities, features)


def getFeatures(NUMPOINTS, NUMCLASS, k):  #k is already the number of clusters

    POINTSPERCLASS = NUMPOINTS / NUMCLASS

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
    remainingPoints = NUMPOINTS - POINTSPERCLASS * NUMCLASS
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


    return  (features,isFull,classId,centers)

