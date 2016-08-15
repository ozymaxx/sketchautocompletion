from scipy.cluster.vq import *

import numpy as np
import random
import copy
from random import shuffle
import scipy.cluster.vq.kmeans2
class fastCKMeans:
    def __init__(self, features, isFull, classId, k, maxiter = 20, votefreq = 2, thres = 10**-10):
        self.features = features
        self.k = k
        self.isFull = isFull
        self.classId = classId
        self.numclass = len(set(self.classId))
        self.clusterFeatures = [[] for i in range(self.k)]
        self.clusterCenters = [[0]*len(features[0])]*k
        self.featureCluster = [0]*len(features)
        self.fullIndex = [idx for idx in range(len(features)) if isFull[idx]]
        self.classvotes = [[0]*k for i in range(max(classId)+1)]
        self.class2cluster = [0]*self.numclass

        self.maxiter = maxiter
        self.thres = thres
        # init clusters
        self.initClusters()

    def initClusters(self, rand=True):
        self.clusterCenters = copy.copy(random.sample(self.features, self.k))

    def getCKMeans(self):
        for curriter in range(self.maxiter):
            kmeans2(self.features,  )

            print 'FastCKMeans iteration %i (max %i)' % (curriter, self.maxiter)
            print 'Assigning features to clusters'
            self.feature2cluster(self.features, self.clusterFeatures, self.classId)
            print 'Calculating cluster centers'
            dist = self.clusterMove(self.features, self.clusterFeatures, self.clusterCenters)

            #if dist < self.thres:len(0
            #    break

        return [self.clusterFeatures, self.clusterCenters]

    def clusterMove(self, features, clusterFeatures, clusterCenters):
        distsum = 0
        for cIdx in range(len(clusterFeatures)):
            featuresum = [0] * len(features[0])
            numFeaturesInCluster = len(clusterFeatures[cIdx])
            for f in clusterFeatures[cIdx]:
                featuresum = [self.features[f][idx] + featuresum[idx] for idx in range(len(self.features[0]))]
            featuresum = [featuresum[idx]/numFeaturesInCluster for idx in range(len(featuresum))]

            # assign the new cluster center
            clusterCenters[cIdx] = featuresum
        return distsum

    def feature2cluster(self, features, clusterFeatures, classId):
        self.clusterFeatures = [[] for i in range(self.k)]
        self.classvotes = [[0]*self.k for i in range(max(classId)+1)]

        for fIdx in range(len(features)):
            _, cc = self.closestClusterCenter(features[fIdx], self.clusterCenters)
            if self.isFull[fIdx]:
                self.classvotes[classId[fIdx]][cc] += 1
            else:
                self.clusterFeatures[cc].append(fIdx)

        # now honour the voting
        print 'Voting'
        for classIdx in range(len(self.classvotes)):
            currclassvotes = self.classvotes[classIdx]
            votedCluster = currclassvotes.index(max(currclassvotes))
            self.class2cluster[classIdx] = votedCluster

            for i in range(self.numclass):
                # so that it cannot be selected by another class
                self.classvotes[i][votedCluster] = -1

        # now voting is done, send full sketches to corresponding
        # clusters
        for fullidx in self.fullIndex:
            clusterforfullidx = self.class2cluster[classId[fullidx]]
            self.clusterFeatures[clusterforfullidx].append(fullidx)

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
        return scipy.spatial.distance.euclidean(x, y)
        #return sum([(x_ - y_)*(x_ - y_) for x_, y_ in zip(x, y)])