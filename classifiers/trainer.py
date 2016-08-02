"""
Classifier trainer
Ahmet BAGLAN - Arda ICMEZ - Semih GUNEL
14.07.2016
"""
import sys
sys.path.append("../../libsvm-3.21/python/")
from svmutil import *
import copy
import numpy as np
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
        Outputs : Models
        """
        label = copy.copy(self.classId)
        order = 0
        allModels = list()
        for i in clusterIdArr:#Foreach cluster
            y = []
            x = []
            for j in i:#Foreach instance in the cluster
                j = int(j)
                y.append(label[j])
                x.append(self.featArr[j].tolist())

            prob  = svm_problem(y, x)
            param = svm_parameter('-s 0 -t 2 -g 0.125 -c 8 -b 1 -q')
            
            m = svm_train(prob, param,)
            allModels.append(m.get_SV())

            import os
            if not os.path.exists(directory):
                os.mkdir(directory)

            svm_save_model(directory + "/" + "clus" + `order` + '.model', m)#Save the model for the cluster
            order+=1
        return allModels


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