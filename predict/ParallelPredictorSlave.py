"""
Predictor
Ahmet BAGLAN
"""
import sys
sys.path.append('../classifiers')
sys.path.append("../../libsvm-3.21/python/")
sys.path.append('../data/')
import numpy as np
import math

from svmutil import *
from trainer import *
from shapecreator import *
from FeatureExtractor import *
from scipy.spatial import distance

class ParallelPredictorSlave:
    """The Slave predictor class
    The main difference between Predictor and ParallelPredictorSlave is that calculatePosterior gives output by name"""
    def __init__(self, kmeansoutput = None, classId = None, subDirectory = None, file = None, svm = None):
        self.kmeansoutput = kmeansoutput
        self.classId = classId
        self.subDirectory = subDirectory
        self.files = file
        self.svm = svm

    def getDistance(self, x, y):
        """Computes euclidian distance between x instance and y instance
        inputs: x,y instances
        kmeansoutput: distance"""
        return distance.euclidean(x, y)

    def clusterProb(self, instance, priorProb):
        """
        Returns P(C_k|x) for each cluster k as a list
        :param instance: instance to get posterior cluster probabiltiies
        :param priorProb: prior cluster probabilities, before seeing the instance to predict
        :return: dictionary of cluster probs for each cluster in a dictionary, normalized
        """
        centers = self.kmeansoutput[1]
        dist = [self.getDistance(instance, centers[idx]) for idx in range(len(centers))]

        sigma = 0.3
        mindist = min(dist)
        diste_ = [math.exp(-1*abs(d - mindist)/(2*sigma*sigma)) for d in dist]
        clustProb = [diste_[idx]*priorProb[idx] for idx in range(len(centers))]

        # normalize
        clustProb = [c/sum(clustProb) for c in clustProb]

        return clustProb

    def svmProb(self, model, instance):
        """Predicts the probability of the given model"""

        ####Prevent Writing----------------------------------------
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
        ### Prevent printing ended-----------------------------------

        return (p_label, p_val)

    def predictIt(self, instance):
        # find the probability of given feature to belong any of athe classes
        priorClusterProb = self.calculatePriorProb()
        classProb = self.calculatePosteriorProb(instance, priorClusterProb)
        return classProb

    def calculatePosteriorProb(self, instance, priorClusterProb, numericKeys = False):
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

        clusterPrb  = self.clusterProb(instance, priorClusterProb)#Probability list to be in a cluster

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
                classesInCluster = [self.classId[int(self.kmeansoutput[0][clstrid][0])]]

            elif clstrid in heteClstrId:
                if not self.svm:
                    modelName = self.subDirectory + "/clus" + str(heteClstrId.index(clstrid)) + ".model"
                    m = svm_load_model(modelName)
                    classesInCluster = m.get_labels()
                    labels, probs = self.svmProb(m, [instance.tolist()])
                else:
                    labels, _, probs = self.svm.predict(int(heteClstrId.index(clstrid)), [instance.tolist()])
                    classesInCluster = self.svm.getlabels(heteClstrId.index(clstrid))

            for c in range(len(classesInCluster)):
                probabilityToBeInThatClass = 1 if clstrid in homoClstrId else probs[0][c]
                outDict[int(classesInCluster[c])] += probabilityToBeInThatCluster * probabilityToBeInThatClass

        if not numericKeys:
            output = {}
            for i in outDict.keys():
                output[self.files[i]] = outDict[i]
            return output

        return outDict

    def calculatePriorProb(self):
        """
        Returns prior probabilities list of being in a cluster, before seeing instance to be predicted
        :return: p = len(cluster)/len(total) for each cluster
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



