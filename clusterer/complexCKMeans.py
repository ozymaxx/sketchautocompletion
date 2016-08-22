from scipy.cluster.vq import *

import numpy as np
import random
import copy
from random import shuffle
from  scipy.cluster.vq import *
import operator
from random import randint
import scipy


class complexCKMeans:
    def __init__(self, features, isFull, classId, k, maxiter=20, votefreq=3, initweight=0, stepweight=0.2):
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
        self.currweight = initweight
        self.maxweight = 2
        self.stepweight = stepweight
        self.initClusters()

    def initClusters(self):
        # randomly select from features
        #self.clusterCenters = copy.copy(random.sample(self.features, self.k))
        print 'Initializing Cluster Centers'
        numFeatureAddedForCluster = [0]*self.k
        # initialize numclass of k cluster to be the center of the full sketches
        if self.numclass <= self.k:
            for fidx in self.fullIndex:
                fclass = self.classId[fidx]
                if fclass < self.k:
                    # add each full sketch to the corresponding cluster, then divide
                    self.clusterCenters[fclass] = map(operator.add,
                                                                  self.clusterCenters[fclass],
                                                                  self.features[fidx])
                    numFeatureAddedForCluster[fclass] += 1

            for clusterCenterIdx in range(self.numclass):
                self.clusterCenters[clusterCenterIdx] = [cfloat/numFeatureAddedForCluster[clusterCenterIdx] for cfloat in self.clusterCenters[clusterCenterIdx]]

        # for the remaining cluster centers, randomly select from the non-selected features
        numClustSelected = self.numclass
        while numClustSelected < self.k:
            featIdx = randint(0, len(self.features))
            if not self.isFull[featIdx]:
                self.clusterCenters[numClustSelected] = self.features[featIdx]
                numClustSelected+=1

    def vote(self, votedclass, votedcluster):
        #print '%i_%i'%(votedclass,votedcluster)
        for clssIdx in range(self.numclass):
            for clstrIdx in range(self.k):
                # mustLinkDistances
                if clssIdx == votedclass and clstrIdx != votedcluster:
                    self.classvotes[clssIdx][clstrIdx] += 1
                # cannotLinkDistances
                if clssIdx != votedclass and clstrIdx == votedcluster:
                    self.classvotes[clssIdx][clstrIdx] += 1

    def getCKMeans(self):
        for iter in range(self.maxiter):
            print 'Running iteration %i (max %i)' % (iter, self.maxiter)
            self.clusterFeatures = [[] for i in range(self.k)]  # features in cluster
            self.classvotes = [[0] * self.k for i in
                               range(max(self.classId) + 1)]  # votes of the classes for cluster, row for class

            from multiprocessing import Pool
            import itertools

            noLabelChange = True
            print 'Calculate distance for each feature to each cluster'
            # calculate distance for each feature to each cluster
            for fidx, f in enumerate(self.features):
                closestcluster = 0
                ccdist = self.euclidiandist(f, self.clusterCenters[0])
                for cidx in range(self.k):
                    newdist = self.euclidiandist(f, self.clusterCenters[cidx])
                    self.featureClusterDist[fidx][cidx] = newdist
                    if newdist < ccdist:
                        closestcluster = cidx
                        ccdist = newdist

                self.vote(self.classId[fidx], closestcluster)

                # if partial, then directly sent it to the closest cluster
                if not self.isFull[fidx]:
                    if self.featureCluster[fidx] != closestcluster:
                        noLabelChange = False
                    self.clusterFeatures[closestcluster].append(fidx)
                    self.featureCluster[fidx] = closestcluster

            # multiply votes by weight
            for voteidx, _ in enumerate(self.classvotes):
                self.classvotes[voteidx] = [self.currweight*v for v in self.classvotes[voteidx]]

            # Honor the voting
            print 'Assing full features to clusters'
            for fidx in self.fullIndex:
                fclass = self.classId[fidx]

                # iterate over each cluster and find the smallest distances
                bestdist, bestclstr = self.featureClusterDist[fidx][0] + self.classvotes[self.classId[fidx]][0], 0
                for clstridx in range(self.k):
                    dist = self.featureClusterDist[fidx][clstridx] + self.classvotes[self.classId[fidx]][clstridx]
                    if dist < bestdist:
                        bestdist = dist
                        bestclstr = clstridx

                if self.featureCluster[fidx] != bestclstr:
                    noLabelChange = False

                self.featureCluster[fidx] = bestclstr
                self.clusterFeatures[bestclstr].append(fidx)

            if noLabelChange:
                print 'Break for no label change'
                break

            print 'Move Cluster Centers'
            self.clusterMove2(self.features, self.clusterFeatures, self.clusterCenters)
            self.currweight = min(self.currweight + self.stepweight, self.maxweight)

        return [self.clusterFeatures, self.clusterCenters]

    def clusterMove(self, features, clusterFeatures, clusterCenters):
        distsum = 0

        for clstrIdx in range(len(clusterFeatures)):
            featuresum = [0] * len(features[0])
            numFeaturesInCluster = len(clusterFeatures[clstrIdx])
            if numFeaturesInCluster != 0:
                for f in clusterFeatures[clstrIdx]:
                    featuresum = [self.features[f][idx] + featuresum[idx] for idx in range(len(self.features[0]))]

                featuresum = [featuresum[idx]/numFeaturesInCluster for idx in range(len(featuresum))]
                # assign the new cluster center
                clusterCenters[clstrIdx] = featuresum
            # what to do with empty cluster
        return distsum


    def clusterMove2(self, features, clusterFeatures, clusterCenters):
        has_members = []
        for i in np.arange(len(clusterFeatures)):
            #cell_members = np.compress(np.equal(self.featureCluster, i), features, 0)
            cell_members = [features[idx] for idx in self.clusterFeatures[i]]
            if len(cell_members) > 0:
                self.clusterCenters[i] = np.mean(cell_members, 0)
                has_members.append(i)
        # remove code_books that didn't have any members
        #self.clusterCenters[i] = np.take(self.clusterCenters[i], has_members, 0)


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
        #return sum([(x[idx]-y[idx])*(x[idx]-y[idx]) for idx in range(len(x))])
        return scipy.spatial.distance.euclidean(x, y)**2
        #return sum([(x_ - y_)*(x_ - y_) for x_, y_ in zip(x, y)])