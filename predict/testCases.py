import numpy as np
import sys

print sys.path
import getConstraints
from ckmeans import *
from trainer import *

def testIt(NUMPOINTS, NUMCLASS, k):  #k is already the number of clusters
    (features, isFull, classId, centers) = getFeatures(NUMPOINTS,NUMCLASS)
    test = getConstraints(NUMPOINTS, isFull, classId)
    kmeans = CKMeans(test, features, k)

    output = kmeans.getCKMeans()
    probabilities = computeProb(output)
    return (output,probabilities, features, classId)



