import sys
import getConstraints
from ckmeans import *
from trainer import *


def testIt(NUMPOINTS, NUMCLASS, k):  #k is already the number of clusters
    (features, isFull, classId, centers) = getFeatures(NUMPOINTS, NUMCLASS)
    test = getConstraints(NUMPOINTS, isFull, classId)
    kmeans = CKMeans(test, features, k)

    kmeansoutput = kmeans.getCKMeans()
    probabilities = computeProb(kmeansoutput)
    return kmeansoutput, probabilities, features, classId



