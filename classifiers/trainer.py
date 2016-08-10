"""
Classifier trainer
Ahmet BAGLAN - Arda ICMEZ - Semih GUNEL
14.07.2016
"""
import sys
sys.path.append("../SVM/")
import copy
import numpy as np
from SVM import *

class Trainer:
    """Trainer Class used for the training"""
    
    def __init__(self, kmeansoutput, classId, featArr = np.array([])):
        # output[0] : list of clusters
        # output[1] : list of cluster centers
        self.featArr = featArr
        self.kmeansoutput = kmeansoutput
        self.classId = classId

    def trainSVM(self, clusterIdArr, directory):
        """Trains the support vector machine and saves models
        Inputs : clusterIdArr ---> clusters list
        Outputs : Support Vectors
        """
        self.svm = SVM(self.kmeansoutput, self.classId, directory, self.featArr)
        self.svm.trainSVM(clusterIdArr, directory)
        return self.svm

    def getHeterogenous(self):
        """
        Gets clusters which are heterogenous
        kmeansoutput :(heterogenousClusters,heterogenousClusterId) -> heterogenous clusters, id's of heterougenous clusters
        """
        heterogenousClusters = list()
        heterogenousClusterId = list()
        for clusterId in range(len(self.kmeansoutput[0])):
            # if class id of any that in cluster of clusterId is any different than the first one
            if any(x for x in range(len(self.kmeansoutput[0][clusterId])) if
                   self.classId[int(self.kmeansoutput[0][clusterId][0])] != self.classId[
                       int(self.kmeansoutput[0][clusterId][x])]):
                heterogenousClusters.append(self.kmeansoutput[0][clusterId])
                heterogenousClusterId.append(clusterId)
        return heterogenousClusters, heterogenousClusterId

    def getHomogenous(self):
        """
        Gets clusters which are homogenous
        kmeansoutput: (homoCluster,homoIdClus) -> homogenous clusters, id's of homogenous clusters
        """
        homoCluster = list()
        homoIdClus = list()
        for clusterId in range(len(self.kmeansoutput[0])):
            # if class id of any that in cluster of clusterId is any different than the first one
            if not any(x for x in range(len(self.kmeansoutput[0][clusterId])) if
                       self.classId[int(self.kmeansoutput[0][clusterId][0])] != self.classId[
                           int(self.kmeansoutput[0][clusterId][x])]):
                homoCluster.append(self.kmeansoutput[0][clusterId])
                homoIdClus.append(clusterId)
        return homoCluster, homoIdClus

    @staticmethod
    def loadSvm(kmeansoutput, classId, subDirectory, featArr):
        svm = SVM(kmeansoutput, classId, subDirectory, featArr)
        return svm
