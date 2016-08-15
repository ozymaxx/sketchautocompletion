import numpy as np
import random
import copy
import scipy
class fastCKMeans:
    def __init__(self, features, isFull, classId, k, maxiter = 20, thres = 10**-10):
        self.features = np.transpose(features)
        self.k = k
        self.isFull = isFull
        self.classId = classId
        self.clusterFeatures = [np.array([], dtype=int) for i in range(self.k)]
        self.clusterCenters = [None]*k
        self.featureCluster = [None]*len(features)
        self.fullIndex = [idx for idx in range(len(features)) if isFull[idx]]
        self.classvotes = [[None]*k for i in range(max(classId)+1)]

        self.maxiter = maxiter
        self.thres = thres
        # init clusters
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
            self.feature2cluster(self.features, self.clusterFeatures, self.classId)
            dist = self.clusterMove(self.features, self.clusterFeatures, self.clusterCenters)

            if dist < self.thres:
                break

        return [self.clusterList, self.centerList]

    def clusterMove(self, features, clusterFeatures, clusterCenters):
        distsum = 0
        for cIdx in range(len(clusterFeatures)):
            featuresum = [None] * len(features[0])
            numFeaturesInCluster = len(clusterFeatures[cIdx])
            for f in clusterFeatures[cIdx]:
                featuresum = [f[idx] + featuresum[idx] for idx in range(len(f))]
            featuresum = [featuresum[idx]/numFeaturesInCluster for idx in range(len(featuresum))]

            # assign the new cluster center
            clusterCenters[cIdx] = featuresum
        return distsum

    def feature2cluster(self, features, clusterFeatures, classId):
        self.classvotes = [[None]*self.k for i in range(max(classId)+1)]
        for fIdx in range(len(features)):
            _, cc = self.closestClusterCenter(features[fIdx], self.clusterCenters)
            if self.isFull[fIdx]:
                self.classvotes[classId[fIdx]][cc] += 1
                pass
            else:
                clusterFeatures[cc].append(fIdx)

        # now honour the voting
        for fIdx in range(len(features)):


    def closestClusterCenter(self, f, clusterCenters):
        smldist = self.euclidiandist(f, clusterCenters[0])
        cc = 0

        for cIdx in range(len(clusterCenters)):
            dist = self.euclidiandist(f, clusterCenters[cIdx])
            if dist < smldist:
                smldist = dist
                cc = cIdx

        return smldist, cc
    def euclidiandist(self, x, y):
        # we might get rid of square root
        return scipy.spatial.distance.euclidean(x,y)