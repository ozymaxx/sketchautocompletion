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
    
    def __init__(self, output, classId, featArr = np.array([]), labArr = np.array([])):
        # output[0] : list of clusters
        # output[1] : list of cluster centers
        self.featArr = featArr
        self.labArr = labArr
        self.output = output
        self.classId = classId
        
    def trainSVM(self):
        # Trains the support vector machine
        label = copy.copy(self.labArr)
        order = 0
        allModels = list()
        for i in self.output[0]:
            y = []
            x = []
            for j in i:
                j = int(j)
                y.append(label[j])
                x.append(self.featArr[j].tolist())
    
            print y, x
            prob  = svm_problem(y, x)
            param = svm_parameter('-s 0 -t 2 -g 0.125 -c 8 -b 1 -q')
            
            m = svm_train(prob, param,)
            allModels.append(m.get_SV())
            svm_save_model('../classifiers/clus' + `order` + '.model', m)
            order+=1
    
        return allModels
    
    def computeProb(self):
        # Returns probability list of being in a cluster
        prob = []
        total =  0
    
        for c in self.output[0]:
            total += len(c)
    
        for cluster in self.output[0]:
            prob.append(len(cluster)/float(total))
        return prob
    
    def getHeterogenous(self):
        # Gets clusters which are heterogenous
        clustersToBeTrained = list()
        clustersIdsToBeTrained = list()
        for clusterId in range(len(self.output[0])):
            # if class id of any that in cluster of clusterId is any different than the first one
            if any(x for x in range(len(self.output[0][clusterId])) if self.classId[int(self.output[0][clusterId][0])] != self.classId[int(self.output[0][clusterId][x])]):
                clustersToBeTrained.append(self.output[0][clusterId])
                clustersIdsToBeTrained.append(clusterId)
        return clustersToBeTrained, clustersIdsToBeTrained
    
    def getHomogenous(self):
        # Gets clusters which are homogenous
        homoClass = list()
        homoIdClus = list()
        for clusterId in range(len(self.output[0])):
            # if class id of any that in cluster of clusterId is any different than the first one
            if not any(x for x in range(len(self.output[0][clusterId])) if self.classId[int(self.output[0][clusterId][0])] != self.classId[int(self.output[0][clusterId][x])]):
                homoClass.append(self.output[0][clusterId])
                homoIdClus.append(clusterId)
        return homoClass, homoIdClus



