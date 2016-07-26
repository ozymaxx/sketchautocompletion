import sys
sys.path.append('../classifiers')
sys.path.append("../../libsvm-3.21/python/")

import numpy as np
import math

from svmutil import *
from trainer import *


class Predictor:
    
    def __init__(self, output, classId):
        self.output = output
        self.classId = classId
        
    def computeDistance(self, x,y):
        # Computes euclidian distance between x instance and y instance
        x = np.asarray(x)
        y = np.asarray(y)
        return np.sqrt(np.sum((x-y)**2))
    
    def clusterProb(self,instance,normalProb):
        # Returns P(Ck|x)
        # normalProb : probability that was calculated in trainer
        probTup = []
        for i in range(len(self.output[1])):
    
            dist = self.computeDistance(instance, self.output[1][i])
            probTup.append(math.exp(-1*abs(dist))*normalProb[i])
        return probTup
    
    def svmProb(self,model,instance):
        # Predicts the probability of the given model
        y = [0]
        p_label, p_acc, p_val = svm_predict(y, instance, model, '-b 1')
        return (p_label, p_val)

    def calculateProb(self, instance, priorClusterProb):#It would increase performance to get list of instance!!!!!!!!!!!!!!!!
        #features : feature array
        #output : list [ List of Cluster nparray, List of Cluster Center nparray]
        #probability : P(Ck)
        # Returns P(Si|x)
    
        # dict of probabilities of given instance belonging to every possible class
        # initially zero
        outDict = dict.fromkeys([i for i in set(self.classId)], 0.0)
        
        predictTrainer = Trainer(self.output,self.classId)
        homoClstrFeatureId, homoClstrId = predictTrainer.getHomogenous()
        heteClstrFeatureId, heteClstrId = predictTrainer.getHeterogenous()
        
        clusterPrb  = self.clusterProb( instance, priorClusterProb)
        
        # normalize cluster probability to add up to 1
        clusterPrb = [x/sum(clusterPrb) for x in clusterPrb]
    
        for clstrid in range(len(self.output[0])):
            probabilityToBeInThatCluster = clusterPrb[clstrid]
            if clstrid in homoClstrId:
                
                # if homogeneous cluster is empty, then do not
                # process it and continue
                if len(self.output[0][clstrid]) == 0:
                    continue
                
                # if homogeneous then only a single class which is the first
                # feature points class
                classesInCluster = [self.classId[self.output[0][clstrid][0]]]
                
            elif clstrid in heteClstrId:
                modelName = "clus"+ ` heteClstrId.index(clstrid) ` +".model"
                m = svm_load_model('../classifiers/' + modelName)
                classesInCluster = m.get_labels()
                labels, probs = self.svmProb(m, [instance.tolist()])
                
            for c in range(len(classesInCluster)):
                probabilityToBeInThatClass = 1 if clstrid in homoClstrId else probs[0][c]
                outDict[int(classesInCluster[c])] += probabilityToBeInThatCluster * probabilityToBeInThatClass
    
        return outDict