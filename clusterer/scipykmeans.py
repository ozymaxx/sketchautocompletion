from scipy.cluster.vq import *

import numpy as np
import random
import copy
from random import shuffle
from scipy.cluster.vq import kmeans2
class scipykmeans:
    def __init__(self, features, isFull, classId, k, maxiter = 20, votefreq = 1, thres = 10**-10):
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
        #self.clusterCenters = copy.copy(random.sample(self.features, self.k))
        pass
    def getCKMeans(self):
        print 'Start k-means'
        '''
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
        '''

        self.clusterCenters, label = kmeans2(np.asarray(self.features), k=self.numclass)

        for lidx in range(len(label)):
            self.clusterFeatures[label[lidx]].append(lidx)


        print 'End k-means'
        return [self.clusterFeatures, self.clusterCenters]

