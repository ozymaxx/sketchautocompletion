from scipy.cluster.vq import *
import numpy as np
import random
import copy
from random import shuffle
from scipy.cluster.vq import kmeans2
from scipy.spatial import distance
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

        self.debugMode = True

    def initClusters(self, rand=True):
        #self.clusterCenters = copy.copy(random.sample(self.features, self.k))
        pass

    def getDistance(self, x, y):
        """Computes euclidian distance between x instance and y instance
        inputs: x,y instances
        kmeansoutput: distance"""
        return distance.euclidean(x, y)
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


        self.clusterCenters, label = kmeans2(np.asarray(self.features), iter = 150, k=self.k, minit= 'points')

        for lidx in range(len(label)):
            self.clusterFeatures[label[lidx]].append(lidx)




        print 'End k-means'
        return [self.clusterFeatures, self.clusterCenters]

    def getKMeansIterated(self, iterNum):
        bestDistance = 0
        self.clusterCenters, label = kmeans2(np.asarray(self.features), iter = 150, k=self.k, minit= 'points')
        for lidx in range(len(label)):
            self.clusterFeatures[label[lidx]].append(lidx)

        bestCenters = copy.copy(self.clusterCenters)
        bestFeatures = copy.copy(self.clusterFeatures)
        for i in range(len(self.clusterFeatures)):
                for j in self.clusterFeatures[i]:
                    bestDistance += self.getDistance(j,self.clusterCenters[i])


        for it in range(iterNum):
            self.clusterFeatures = [[] for i in range(self.k)]
            self.clusterCenters = [[0]*len(self.features[0])]*self.k

            self.clusterCenters, label = kmeans2(np.asarray(self.features), iter = 150, k=self.k, minit= 'points')

            for lidx in range(len(label)):
                self.clusterFeatures[label[lidx]].append(lidx)
            total = 0
            for i in range(len(self.clusterFeatures)):
                for j in self.clusterFeatures[i]:
                    total += self.getDistance(j,self.clusterCenters[i])

            changeBest = True
            if(total<bestDistance):
                for cl in self.clusterFeatures:
                    print self.k
                    if(len(cl)< (len(self.features)/self.k)/2):
                        print "There is one group which is tooo weak"
                        changeBest =False

                if changeBest:
                    bestCenters = copy.copy(self.clusterCenters)
                    bestFeatures = copy.copy(self.clusterFeatures)
                    bestDistance = total

            if(self.debugMode):
                print total
        return [bestFeatures, bestCenters]

