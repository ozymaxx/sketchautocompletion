"""
Classifier trainer
an Interface class for training Svm models given kmeans output
Ahmet BAGLAN - Arda ICMEZ - Semih GUNEL
14.07.2016
"""
import sys
sys.path.append("../SVM/")
import copy
import numpy as np
from LibSVM import *

class Trainer:
    def __init__(self, kmeansoutput, classid, features=np.array([])):
        self.kmeansoutput = kmeansoutput  # kmeansoutput=[cluster_feature_ids, cluster_centers]
        self.classid = classid
        self.features = features
        self.svm = None

    def trainSVM(self, clusterFeatureIds, directory):
        """
        trains svm models for the given cluster ids
        :param clusterFeatureIds: clusters to be trained an svm model for
        :param directory: directory for svm models to be saved
        :return: returns the svm class governing svm models
        """
        self.svm = LibSVM(self.kmeansoutput, self.classid, directory, self.features)
        #self.svm.trainSVM(clusterFeatureIds, directory)
        self.svm.trainSVM(clusterFeatureIds, directory)
        print "Training finished"
        return self.svm

    def getHeterogenousClusterId(self):
        """
        Gets clusters ids which contain instances of different classes, therefore heterogenous
        :return: feature ids of cluster heterogenous clusters, list of cluster ids
        """
        heterogenousClusters = list()
        heterogenousClusterId = list()
        for clusterId in range(len(self.kmeansoutput[0])):
            # if class id of any that in cluster of clusterId is any different than the first one
            if any(x for x in range(len(self.kmeansoutput[0][clusterId])) if
                   self.classid[int(self.kmeansoutput[0][clusterId][0])] != self.classid[
                       int(self.kmeansoutput[0][clusterId][x])]):
                heterogenousClusters.append(self.kmeansoutput[0][clusterId])
                heterogenousClusterId.append(clusterId)
        return heterogenousClusters, heterogenousClusterId

    def getHomogenousClusterId(self):
        """
        Gets clusters ids which contain instances of only one classes, therefore homogenous
        :return: feature ids of cluster homogenous clusters, list of cluster ids
        """
        homoCluster = list()
        homoIdClus = list()
        for clusterId in range(len(self.kmeansoutput[0])):
            # if class id of any that in cluster of clusterId is any different than the first one
            if not any(x for x in range(len(self.kmeansoutput[0][clusterId])) if
                       self.classid[int(self.kmeansoutput[0][clusterId][0])] != self.classid[
                           int(self.kmeansoutput[0][clusterId][x])]):
                homoCluster.append(self.kmeansoutput[0][clusterId])
                homoIdClus.append(clusterId)
        return homoCluster, homoIdClus

    @staticmethod
    def loadSvm(kmeansoutput, classid, directory, features):
        svm = LibSVM(kmeansoutput, classid, directory, features)
        return svm
