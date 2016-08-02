"""
Predictor
Ahmet BAGLAN
14.07.2016
"""
import sys
sys.path.append('../classifiers')
sys.path.append("../../libsvm-3.21/python/")

import numpy as np
import math

from svmutil import *
from trainer import *
from shapecreator import *
from FeatureExtractor import *

class newMethodPredictor:
    """The predictor class implementing functions to return probabilities"""
    def __init__(self, kmeansoutput, classId, subDirectory):
        self.kmeansoutput = kmeansoutput
        self.classId = classId
        self.subDirectory = subDirectory

    def getDistance(self, x, y):
        """Computes euclidian distance between x instance and y instance
        inputs: x,y instances
        kmeansoutput: distance"""
        x = np.asarray(x)
        y = np.asarray(y)
        return np.sqrt(np.sum((x-y)**2))

    def clusterProb(self, instance, normalProb):
        """
        Returns P(Ck|x)
        normalProb : probability that was calculated in trainer
        instance: feature list of the instance to be queried"""
        probTup = []
        for i in range(len(self.kmeansoutput[1])):

            dist = self.getDistance(instance, self.kmeansoutput[1][i])
            probTup.append(math.exp(-1*abs(dist))*normalProb[i])
        return probTup

    def svmProb(self,model,instance):
        """Predicts the probability of the given model"""
        ###Shittty code ---------------->
        ####Prevent Writing
        import sys
        class NullWriter(object):
            def write(self, arg):
                pass
        nullwrite = NullWriter()
        oldstdout = sys.stdout
        sys.stdout = nullwrite # disable kmeansoutput

        y = [0]
        p_label, p_acc, p_val = svm_predict(y, instance, model, '-b 1')

        sys.stdout = oldstdout # enable kmeansoutput
        ### Prevent printing ended

        return (p_label, p_val)

    def predictIt(self, instance):
        # find the probability of given feature to belong any of athe classes
        priorClusterProb = self.computePriorProb()
        classProb = self.calculatePosteriorProb(instance, priorClusterProb)
        return classProb

    def predictByPath(self, fullsketchpath):
        instance = featureExtract(fullsketchpath)
        priorClusterProb = self.computePriorProb()
        classProb = predictor.calculatePosteriorProb(instance, priorClusterProb)
        return classProb

    def predictByString(self, jstring):
        loadedSketch = shapecreator.buildSketch('json', jstring)
        featextractor = IDMFeatureExtractor()
        instance = featextractor.extract(loadedSketch)
        priorClusterProb = self.computePriorProb()
        classProb = self.calculatePosteriorProb(instance, priorClusterProb)
        return classProb

    def calculatePosteriorProb(self, instance, priorClusterProb):
        """
        #features : feature array
        #kmeansoutput : list [ List of Cluster nparray, List of Cluster Center nparray]
        #p  robability : P(Ck)
        # Returns P(Si|x)
        """
        # dict of probabilities of given instance belonging to every possible class
        # initially zero
        outDict = dict.fromkeys([i for i in set(self.classId)], 0.0)

        homoClstrFeatureId, homoClstrId = self.getHomogenous()
        heteClstrFeatureId, heteClstrId = self.getHeterogenous()

        clusterPrb  = self.clusterProb( instance, priorClusterProb)#Probability list to be in a cluster

        # normalize cluster probability to add up to 1
        clusterPrb = [x/sum(clusterPrb) for x in clusterPrb]

        for clstrid in range(len(self.kmeansoutput[0])):
            probabilityToBeInThatCluster = clusterPrb[clstrid]
            if clstrid in homoClstrId:

                # if homogeneous cluster is empty, then do not
                # process it and continue
                if len(self.kmeansoutput[0][clstrid]) == 0:
                    continue

                # if homogeneous then only a single class which is the first
                # feature points class
                classesInCluster = [self.classId[self.kmeansoutput[0][clstrid][0]]]

            elif clstrid in heteClstrId:
                modelName = self.subDirectory +"/clus" + ` heteClstrId.index(clstrid) ` +".model"
                m = svm_load_model(modelName)
                classesInCluster = m.get_labels()
                labels, probs = self.svmProb(m, [instance.tolist()])

            for c in range(len(classesInCluster)):
                probabilityToBeInThatClass = 1 if clstrid in homoClstrId else probs[0][c]
                outDict[int(classesInCluster[c])] += probabilityToBeInThatCluster * probabilityToBeInThatClass
        return outDict

    def computePriorProb(self):
        """
        Returns prior probabilities list of being in a cluster
        p = len(cluster)/len(total)
        """
        prob = []
        total = 0

        for c in self.kmeansoutput[0]:
            total += len(c)

        for cluster in self.kmeansoutput[0]:
            prob.append(len(cluster) / float(total))
        return prob


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
                   self.classId[int(self.kmeansoutput[0][clusterId][0])] != self.classId[int(self.kmeansoutput[0][clusterId][x])]):
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
                       self.classId[int(self.kmeansoutput[0][clusterId][0])] != self.classId[int(self.kmeansoutput[0][clusterId][x])]):
                homoCluster.append(self.kmeansoutput[0][clusterId])
                homoIdClus.append(clusterId)
        return homoCluster, homoIdClus