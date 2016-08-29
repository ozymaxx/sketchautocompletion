"""
Constrained K-Means Implementation in pure python, without cuda support.
Following implementation runs k-means while exploiting background knowledge for full sketches,
so that instances are encouggrated to fall into the same cluster with the majority of their classes,
however they are not forced to. The middle ground is between negative voting of each full sketches in
the following manner. When a full sketch is assigned, they cast following votes

- A Negative vote for full sketches of the same class to be assigned to other clusters
- A Negative vote for full sketches of the other classes to be assigned to the same cluster

When the time for assigning a cluster has arrived, a full sketch looks for each cluster,
considers the negative votes together with distance to each individual cluster, and applies
a weighted some of two vectors. Eventually it gets assigned to a cluster with smallest
sum. Partial sketches assigned in a simmilar manner with the k-means without constraints.
So that they do not consider negative voted but only the distance for each cluster center.

Weight parameters for balancing of negative votes and distance metric.
the total sum is calculated as (weight*number_of_negative_votes + distance)
so that weight = 0 implies k-means without constraints, and a large weight
implies negletion of distances, and only total voting for each class. Therefore
significatly increases the change of all full sketches of the same class will
be assigned to the same cluster

Implementation is taken from MATLAB code written by Caglar Tirkaz
For more information read https://www.cs.cornell.edu/home/cardie/papers/icml-2001.ps
"""

from scipy.cluster.vq import *
import numpy as np
import random
import copy
from random import shuffle
from scipy.cluster.vq import *
import operator
from random import randint
import scipy


class ComplexCKMeans:
    def __init__(self, features, isfull, classid, k, maxiter=20, votefreq=3, initweight=0, stepweight=0.2, maxweight=2):
        self.features = features
        self.k = k
        self.isFull = isfull
        self.classid = classid
        self.numclass = len(set(self.classid))
        self.clusterFeatures = [[] for i in range(self.k)]  # features in cluster
        self.clusterCenters = [[0]*len(features[0])]*k  # centers of clusters
        self.featureCluster = [0]*len(features)  # which cluster a feature belongs to
        self.fullIndex = [idx for idx in range(len(features)) if isfull[idx]]  # index of full sketches
        self.classvotes = [[0] * k for i in range(max(classid) + 1)]  # votes of the classes for cluster, row for class
        # think class votes a matrix having rows as classes and columns as clusters, then each feature votes -for
        # its corresponding class-
        self.class2cluster = [0]*self.numclass # assigned class to cluster
        self.featureClusterDist = [[0]*self.k for i in range(len(self.features))]  # distance of feature to cluster

        self.maxiter = maxiter
        self.votefreq = votefreq  # after which k-means iteration voting should be done
        self.initweight = initweight
        self.currweight = initweight
        self.maxweight = maxweight
        self.stepweight = stepweight

        self.initClusters()

    def initClusters(self):
        """
        For each different class, the center of the full sketches is calculated which is set to be the center
        of the initial cluster. For the remaining cluster centers a random partial symbol is selected from the
        remaining features, which are then serve as the initial cluster centers.
        """
        print 'Initializing Cluster Centers'
        numFeatureAddedForCluster = [0]*self.k
        # initialize numclass of -out of-k cluster to be the center of the full sketches
        if self.numclass <= self.k:
            for fidx in self.fullIndex:
                fclass = self.classid[fidx]
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
                numClustSelected += 1

    def vote2(self, votedclass, votedcluster):
        #print '%i_%i'%(votedclass,votedcluster)
        for clssIdx in range(self.numclass):
            for clstrIdx in range(self.k):
                # mustLinkDistances
                if clssIdx == votedclass and clstrIdx != votedcluster:
                    self.classvotes[clssIdx][clstrIdx] += 1
                # cannotLinkDistances
                if clssIdx != votedclass and clstrIdx == votedcluster:
                    self.classvotes[clssIdx][clstrIdx] += 1

    def vote(self, votingclass, votedcluster):
        """
        Voting can be considered as the negative votes. Each full sketch votes for
        to be not assigned to -to its own class and as well as other classes-. Read class header
        for more information
        """
        for clstrIdx in range(self.k):
            # vote to prevent other classes to assign to the same cluster
            self.classvotes[votingclass][clstrIdx] += 1
        for clssIdx in range(self.numclass):
            # vote to prevent voting class to assign other clusters
            self.classvotes[clssIdx][votedcluster] += 1

            # remove the redundancy
        self.classvotes[votingclass][votedcluster] -= 2

    def getCKMeans(self):
        for iter in range(self.maxiter):
            print 'Running iteration %i (max %i)' % (iter, self.maxiter)
            self.clusterFeatures = [[] for i in range(self.k)]  # features in cluster
            self.classvotes = [[0] * self.k for i in
                               range(max(self.classid) + 1)]  # votes of the classes for cluster, row for class

            from multiprocessing import Pool
            import itertools

            # if no label change, break before reaching maxiter iterations
            noLabelChange = True
            print 'Calculate distance for each feature to each cluster'
            # calculate distance for each feature to each cluster
            # often bottleneck
            for fidx, f in enumerate(self.features):
                closestcluster = 0
                ccdist = self.euclidiandist(f, self.clusterCenters[0])
                for cidx in range(self.k):
                    newdist = self.euclidiandist(f, self.clusterCenters[cidx])
                    self.featureClusterDist[fidx][cidx] = newdist
                    if newdist < ccdist:
                        closestcluster = cidx
                        ccdist = newdist

                if self.isFull[fidx]:
                    # if full vote
                    self.vote(self.classid[fidx], closestcluster)
                # if partial, then directly sent it to the closest cluster
                elif not self.isFull[fidx]:
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
                fclass = self.classid[fidx]

                # iterate over each cluster and find the smallest distance
                bestdist, bestclstr = self.featureClusterDist[fidx][0] + self.classvotes[self.classid[fidx]][0], 0
                for clstridx in range(self.k):
                    dist = self.featureClusterDist[fidx][clstridx] + self.classvotes[self.classid[fidx]][clstridx]
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
            self.clusterMove(self.features, self.clusterFeatures, self.clusterCenters)
            self.currweight = min(self.currweight + self.stepweight, self.maxweight)

        return [self.clusterFeatures, self.clusterCenters]

    def clusterMove2(self, features, clusterFeatures, clusterCenters):
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

    def clusterMove(self, features, clusterFeatures, clusterCenters):
        """
        Move cluster centers
        :param features: id to 720-dimensional feature vectors
        :param clusterFeatures: feature ids for each cluster
        :param clusterCenters: centers for each cluster
        """
        has_members = []
        for i in np.arange(len(clusterFeatures)):
            #cell_members = np.compress(np.equal(self.featureCluster, i), features, 0)
            cell_members = [features[idx] for idx in self.clusterFeatures[i]]
            if len(cell_members) > 0:
                self.clusterCenters[i] = np.mean(cell_members, 0)
                has_members.append(i)
        # remove code_books that didn't have any members
        #self.clusterCenters[i] = np.take(self.clusterCenters[i], has_members, 0)

    # actually returns euclidian distance sqaured
    # this miss-naming consistent in MATLAB code too
    # therefore I do not change it here
    def euclidiandist(self, x, y):
        # funny that taking square of scipy implementation of euc-dist is faster than
        # pure python implementation of euclidian distance sqaured :(
        return scipy.spatial.distance.euclidean(x, y)**2
