"""
Pedictions for class probabilities, given kmeansoutput -which specifies clusters- and trained svm models.
Predictor is written with the complience as the following paper:
http://iui.ku.edu.tr/sezgin_publications/2012/PR%202012%20Sezgin.pdf
with one possible deviance, which can be seen in the clusterProb method. Although paper
does not present a scaling factor for the feature-to-cluster center distance, putting one,
as have done in the MATLAB code written by Caglar Tirkaz, boosts accuracies significantly.

If predictor is asked for multiple instances at a single time -list of instances-, it uses multiprocess
libraries to speed up the process. This is of course not conceivable in end user experince, although might
get handy while testing.
"""

"""
Predictor
Ahmet BAGLAN, Semih GUNEL
14.07.2016
"""

import math
import numpy as np

from trainer import *
from shapecreator import *
from FeatureExtractor import *
from scipy.spatial import distance

class Predictor:
    def __init__(self, kmeansoutput, classid, directory, svm=None):
        self.kmeansoutput = kmeansoutput
        self.classId = classid
        self.directory = directory  # directory to load svm models
        self.svm = svm

    def euclidiandistance(self, x, y):
        return distance.euclidean(x, y)
    
    def clusterProb(self, instance, priorProb):
        """
        Returns P(C_k|x) for each cluster k as a list
        :param instance: instance to get posterior cluster probabiltiies
        :param priorProb: prior cluster probabilities, before seeing the instance to predict
        :return: dictionary of cluster probs for each cluster in a dictionary, normalized
        """
        centers = self.kmeansoutput[1]
        dist = [self.euclidiandistance(instance, centers[idx]) for idx in range(len(centers))]

        # scaling factor
        sigma = 0.3
        mindist = min(dist)
        diste_ = [math.exp(-1*abs(d - mindist)/(2*sigma*sigma)) for d in dist]
        clustProb = [diste_[idx]*priorProb[idx] for idx in range(len(centers))]

        # normalize
        clustProb = [c/sum(clustProb) for c in clustProb]

        return clustProb

    def predictByInstance(self, instance):
        # find the probability of given feature to belong any of athe classes
        priorClusterProb = self.calculatePriorProb()
        classProb = self.calculatePosteriorProb(instance, priorClusterProb)
        return classProb

    def predictByPath(self, fullsketchpath):
        instance = featureExtract(fullsketchpath)
        priorClusterProb = self.calculatePriorProb()
        classProb = self.calculatePosteriorProb(instance, priorClusterProb)
        return classProb

    def predictByString(self, jstring):
        loadedSketch = shapecreator.buildSketch('json', jstring)
        featextractor = IDMFeatureExtractor()
        instance = featextractor.extract(loadedSketch)

        classProb = self.calculatePosteriorProb(instance, priorClusterProb)
        return classProb
    
    def multi_run_wrapper(self, args):
        """
        ARDA
        :param args:
        :return:
        """
        return self.calculatePosteriorMulti(*args)
    
    def _pickle_method(self, m):
        """
        ARDA
        :param m:
        :return:
        """
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)
    
    def calculatePosteriorProb(self, instances, priorClusterProb):
        """
        Predict labels for instance(s), given P(C_k) for each cluster,
        calculate P(S_i) for each class i

        If the input of list of instances, then a list of class probabilities
        is returned.
        :param instances: instance to be predicted labels for
        :param priorClusterProb: prior cluster probabilities before seeing the data
        :return: dictionary holding probabilities for each class
        """
        if isinstance(instances, list):
            # if asked for multiple predictions, use multiprocessing
            copy_reg.pickle(types.MethodType, self._pickle_method)
            from multiprocessing import Pool
            import itertools
            pool = Pool(4)
            result = pool.map(self.multi_run_wrapper,
                                  [(instances, priorClusterProb, 0),
                                   (instances, priorClusterProb, 1),
                                   (instances, priorClusterProb, 2),
                                   (instances, priorClusterProb, 3)])
            
            super_dict = {}
            for d in result:
                super_dict.update(d)
            super_list = [v for v in super_dict.values()]
            
            return super_list
        else:
            return self.calculatePosteriorSingle(instances, priorClusterProb)
    
    def calculatePosteriorSingle(self, instance, priorClusterProb):
        """
        Predict labels for single instance, given P(C_k) for each cluster,
        calculate P(S_i) for each class i
        :param instance: instance to be predicted labels for
        :param priorClusterProb: prior cluster probabilities before seeing the data
        :return: dictionary holding probabilities for each class
        """

        # dict of probabilities of given instance belonging to every possible class
        # initially zero
        class_prob = dict.fromkeys([i for i in set(self.classId)], 0.0)
        
        homoClstrFeatureId, homoClstrId = self.getHomogenousClusterId()
        heteClstrFeatureId, heteClstrId = self.getHeterogenousClusterId()

        clusterProb = self.clusterProb(instance, priorClusterProb) # Probability list to be in a cluster after seeing instance

        # normalize cluster probability to add up to 1
        clustersum = sum(clusterProb)
        clusterProb = [x/clustersum for x in clusterProb]

        clusterFeatureId = self.kmeansoutput[0]

        for clstrid in range(len(clusterFeatureId)):
            probabilityToBeInThatCluster = clusterProb[clstrid]
            if clstrid in homoClstrId:
                
                # if homogeneous cluster is empty, then do not
                # process it and continue
                if len(clusterFeatureId[clstrid]) == 0:
                    continue
                
                # if homogeneous then only a single class which is the first
                # feature points class
                classesInCluster = [self.classId[int(clusterFeatureId[clstrid][0])]]
                
            elif clstrid in heteClstrId:
                labels, _, svmprobs = self.svm.predict(int(heteClstrId.index(clstrid)), [instance.tolist()])
                classesInCluster = self.svm.getlabels(heteClstrId.index(clstrid))
                
            for clstridx in range(len(classesInCluster)):
                probabilityToBeInThatClass = 1 if clstrid in homoClstrId else svmprobs[0][clstridx]
                class_prob[int(classesInCluster[clstridx])] += probabilityToBeInThatCluster * probabilityToBeInThatClass

        return class_prob
    
    def calculatePosteriorMulti(self, instances, priorClusterProb, procId):
        """
        Predict labels for multiple instances, given P(C_k) for each cluster,
        calculate P(S_i) for each class i
        :param instances: instance to be predicted labels for
        :param priorClusterProb: prior cluster probabilities before seeing the data
        :return: dictionary holding probabilities for each class
        """
        print "Thread",procId, "has started"
        resultDict= {}
        
        homoClstrFeatureId, homoClstrId = self.getHomogenousClusterId()
        heteClstrFeatureId, heteClstrId = self.getHeterogenousClusterId()
        clusterFeatureId = self.kmeansoutput[0]
        
        while procId < len(instances):
            instance = instances[procId]
        
            # dict of probabilities of given instance belonging to every possible class
            # initially zero
            outDict = dict.fromkeys([i for i in set(self.classId)], 0.0)
            
            clusterProb = self.clusterProb(instance, priorClusterProb)#Probability list to be in a cluster
    
            # normalize cluster probability to add up to 1
            clustersum = sum(clusterProb)
            clusterProb = [x/clustersum for x in clusterProb]
    
            for clstrid in range(len(clusterFeatureId)):
                probabilityToBeInThatCluster = clusterProb[clstrid]
                if clstrid in homoClstrId:
                    
                    # if homogeneous cluster is empty, then do not
                    # process it and continue
                    if len(clusterFeatureId[clstrid]) == 0:
                        continue
                    
                    # if homogeneous then only a single class which is the first
                    # feature points class
                    classesInCluster = [self.classId[int(clusterFeatureId[clstrid][0])]]
                    
                elif clstrid in heteClstrId:
                        labels, _, svmprobs = self.svm.predict(int(heteClstrId.index(clstrid)), [instance.tolist()])
                        classesInCluster = self.svm.getlabels(heteClstrId.index(clstrid))
                    
                for clstridx in range(len(classesInCluster)):
                    probabilityToBeInThatClass = 1 if clstrid in homoClstrId else svmprobs[0][clstridx]
                    outDict[int(classesInCluster[clstridx])] += probabilityToBeInThatCluster * probabilityToBeInThatClass
            resultDict[procId] = outDict
            procId += 4

        print "Thread", procId % 4, "has ended"
        return resultDict
             
    def calculatePriorProb(self):
        """
        Returns prior probabilities list of being in a cluster, before seeing instance to be predicted
        :return: p = len(cluster)/len(total) for each cluster
        """
        clusterFeatureId = self.kmeansoutput[0]
        numFeatures = sum([len(cluster) for cluster in clusterFeatureId])

        prob = [len(clusterFeatureId[index]) / float(numFeatures) for index in range(len(clusterFeatureId))]

        return prob

    def getHeterogenousClusterId(self):
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

    def getHomogenousClusterId(self):
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



