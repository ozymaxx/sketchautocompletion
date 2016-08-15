import numpy as np
import random
import copy
class fastCKMeans:
    def __init__(self, features, isFull, classId, k, maxiter = 20):
        self.features = np.transpose(features)
        self.k = k
        self.isFull = isFull
        self.classId = classId
        self.clusterFeatures = [np.array([], dtype=int) for i in range(self.k)]
        self.clusterCenters = [None]*k
        self.maxiter = maxiter
        self.initClusters()

    def initCluster(self, rand=True):
        for i in range(0, self.k):
            # Select unique cluster centers randomly
            featIdx = random.randint(0, self.features.shape[0] - 1)
            while self.features[featIdx] in self.clusterCenters:
                featIdx = random.randint(0, self.features.shape[0] - 1)

            center = copy.copy(self.features[featIdx])
            self.clusterCenters.append(center)

    def getCKMeans(self):
        for curriter in range(self.maxiter):
            print 'FastCKMeans iteration %i (max %i)' % (curriter, self.maxiter)

        


        return self.clusterList, self.centerList

