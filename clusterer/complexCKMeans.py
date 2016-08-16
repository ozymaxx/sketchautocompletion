from scipy.cluster.vq import *

import numpy as np
import random
import copy
from random import shuffle
from  scipy.cluster.vq import *
import operator
from random import randint


class complexCKMeans:
    def __init__(self, features, isFull, classId, k, maxiter = 20, votefreq = 3, initweight = 2):
        self.features = features
        self.k = k
        self.isFull = isFull
        self.classId = classId
        self.numclass = len(set(self.classId))
        self.clusterFeatures = [[] for i in range(self.k)] # features in cluster
        self.clusterCenters = [[0]*len(features[0])]*k # centers of clusters
        self.featureCluster = [0]*len(features) # which cluster a feature belongs to
        self.fullIndex = [idx for idx in range(len(features)) if isFull[idx]] # index of full sketches
        self.classvotes = [[0]*k for i in range(max(classId)+1)] # votes of the classes for cluster, row for class
        self.class2cluster = [0]*self.numclass # assigned class to cluster
        self.featureClusterDist = [[0]*self.k for i in range(len(self.features))] # distance of feature to cluster

        self.maxiter = maxiter
        self.votefreq = votefreq # after which k-means iteration voting should be done
        # init clusters
        self.initweight = initweight
        self.maxweight = 2
        self.stepweight = 0.2
        self.initClusters()

    def initClusters(self):
        # randomly select from features
        #self.clusterCenters = copy.copy(random.sample(self.features, self.k))
        print 'Initializing Cluster Centers'

        # initialize numclass of k cluster to be the center of the full sketches
        if self.numclass < self.k:
            for fidx in self.fullIndex:
                if fidx < self.k:
                    # add each full sketch to the corresponding cluster, then divide
                    self.clusterCenters[self.classId[fidx]] = map(operator.add,
                                                                  self.clusterCenters[self.classId[fidx]],
                                                                  self.features[fidx])

            for clusterCenterIdx in range(len(self.clusterCenters)):
                numFeatureAddedForCluster = self.classId.count(clusterCenterIdx)
                clusterCenter = [cfloat/numFeatureAddedForCluster for cfloat in clusterCenter]

        # for the remaining cluster centers, randomly select from the non-selected features
        numClustSelected = self.numclass
        while numClustSelected < self.k:
            featIdx = randint(0, 9)
            if not self.isFull[featIdx]:
                self.clusterCenter[numClustSelected] = self.features[featIdx]


    def getCKMeans(self):
        previouslabel = [0]*len(self.features)
        label = [0]*len(self.features)
        for iter in range(self.maxiter):
            print 'Running iteration %i (max %i)' % (iter, self.maxiter)
            #self.clusterCenters, label = kmeans2(np.asarray(self.features), k=np.asarray(self.clusterCenters), missing='warn',
            #                                     iter=self.votefreq)

            # calculate distance for each feature to each cluster
            for fidx, f in enumerate(self.features):
                cc = 0
                cdist = self.euclidiandist(f, self.clusterCenters[0])
                for cidx in range(len(self.k)):
                    newdist = self.euclidiandist(f, self.clusterCenters[cidx])
                    self.featureClusterDist[fidx][cidx] = newdist
                    if newdist < cdist:
                        cc = cidx
                        cdist = newdist
                        self.classvotes[self.classId[fidx]][cidx] += 1
                    label[fidx] = cc

                    # if partial, then directly sent it to the closest cluster
                    if not self.isFull[fidx]:
                        self.clusterFeatures.append(fidx)


            # apply anti-votes, each class also votes negatively for other clusters
            for clssidx in range(len(self.classvotes)):



            print 'Running for voting'
            self.classvotes = [[0] * self.k for i in range(max(self.classId) + 1)]
            for fullidx in self.fullIndex:
                self.classvotes[self.classId[fullidx]][label[fullidx]] += 1

            print 'Assign features to clusters'
            self.feature2cluster(self.features, self.clusterFeatures, self.classId, label)
            print 'Move Cluster Centers'
            self.clusterMove(self.features, self.clusterFeatures, self.clusterCenters)


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

    def feature2cluster(self, features, clusterFeatures, classId, label):
        self.clusterFeatures = [[] for i in range(self.k)]

        self.classvotes = [[0]*self.k for i in range(max(classId)+1)]

        for fIdx in range(len(features)):
            cc = label[fIdx]
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