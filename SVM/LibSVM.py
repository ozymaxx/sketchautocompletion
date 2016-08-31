"""
an Interface class for calling LibSVM library -training and prediction- using multiprocessing capabilities of python
Look https://www.csie.ntu.edu.tw/~cjlin/libsvm/ for more information on libsvm and python wrapper we are using
"""

import copy_reg
import types
import copy
import sys
from svmutil import *

class LibSVM:
    def __init__(self, kmeansoutput, classid, directory, features):
        self.kmeansoutput = kmeansoutput
        self.classid = classid  # classes -or labels- of features
        self.directory = directory  # directory to save and load model files
        self.models = {}  # libsvm models for each cluster
        self.features = features
        self.supportVectors = []  # support vecotrs for support vector machines
        
    def multi_run_wrapper(self, args):
        return self.trainMultiCoreSVM(*args)

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

    def doMultiCoreSVM(self, clusterFeatureIds, directory):
        """
        ARDA
        :param clusterFeatureIds:
        :param directory:
        :return:
        """
        copy_reg.pickle(types.MethodType, self._pickle_method)
        from multiprocessing import Pool
        import itertools
        pool = Pool(4)
        pool.map(self.multi_run_wrapper, [(clusterFeatureIds, directory, 0), (clusterFeatureIds, directory, 1),
            (clusterFeatureIds, directory, 2), (clusterFeatureIds, directory, 3)])
        pool.join()
        pool.close()
    def trainSVM(self, clusterFeatureIds, directory):
        """
        Train an support vector machine model for each heterogenous cluster
        :param clusterFeatureId: List of lists having feature ids of clusters
        :param directory: directory to save .model files
        """
        print 'Training SVM with %i features' % sum(len(cluster) for cluster in clusterFeatureIds)
        #label = copy.copy(self.classId) WHY
        order = 0
        for clusterFeatureId in clusterFeatureIds:  # Foreach cluster
            y = []
            x = []
            for featureid in clusterFeatureId:  # Foreach instance in the cluster
                featureid = int(featureid)
                y.append(self.classid[featureid])
                x.append(self.features[featureid].tolist())

            problem = svm_problem(y, x)
            param = svm_parameter('-s 0 -t 2 -g 0.125 -c 8 -b 1 -q')# parameters taken from the MATLAB code written by Caglar Tirkaz

            m = svm_train(problem, param)
            import os
            if directory and not os.path.exists(directory):
                os.mkdir(directory)
            
            if directory:
                svm_save_model(directory + "/" + "clus" + str(order) + '.model', m)  # Save the model for the cluster
                print 'Saved Model %s' % str(directory + "/" + "clus" + str(order) + '.model')
            self.models[order] = m
            order += 1
        print 'Training SVM is done'
        
    def trainMultiCoreSVM(self, clusterIdArr, directory, procId):
        """
        :param clusterFeatureId: List of lists having feature ids of clusters
        :param directory: directory to save .model files
        :return:
        """
        print 'Training SVM with %i features' % sum(len(cluster) for cluster in clusterIdArr)

        #label = copy.copy(self.classId) WHY
        label = self.classid
        order = procId
        while order < len(clusterIdArr):  # Foreach cluster
            clusterFeatureId = clusterIdArr[order]
            y = []
            x = []
            for featureid in clusterFeatureId:  # Foreach instance in the cluster
                featureid = int(featureid)
                y.append(label[featureid])
                x.append(self.features[featureid].tolist())

            problem = svm_problem(y, x)
            param = svm_parameter('-s 0 -t 2 -g 0.125 -c 8 -b 1 -q')
            # parameters taken from the MATLAB code written by Caglar Tirkaz

            m = svm_train(problem, param)
            print order%4, ". thread finished", 1 + (order/4),"models"
            self.supportVectors.extend(m.get_SV())

            import os
            if directory and not os.path.exists(directory):
                os.mkdir(directory)
            
            if directory:
                svm_save_model(directory + "/" + "clus" + str(order) + '.model', m)  # Save the model for the cluster
                print 'Saved Model %s' % str(directory + "/" + "clus" + str(order) + '.model')
            self.models[order] = m
            order += 4
        print 'Training SVM is done'

    def getlabels(self, model_index):
        """
        :param model_index: index of the model
        :return:labels -classes- found in the model
        """
        return self.models[int(model_index)].get_labels()

    def loadModels(self):
        """
        loads model files into memory from the directory given in constructor
        """
        import os
        order = 0
        while os.path.exists(self.directory + "/" + "clus" + str(order) + '.model'):
            print 'Loading SVM Model %s ' % str(self.directory + "/" + "clus" + str(order) + '.model')
            self.models[order] = svm_load_model(self.directory + "/" + "clus" + str(order) + '.model')
            order += 1

    def predict(self, model_index, instance):
        """
        :param model_index: svm model predicting class probabilty
        :param instance: instance to be predicted
        :return: libSVM outputs, p_label, p_acc, p_val; check libsvm readme for more information
        """
        if not self.models:
            order = 0
            import os
            while os.path.exists(self.directory + "/" + "clus" + str(order) + '.model'):
                self.models[order] = svm_load_model(self.directory + "/" + "clus" + str(order) + '.model')
                order += 1

        import sys
        class NullWriter(object):
            def write(self, arg):
                pass
        nullwrite = NullWriter()
        oldstdout = sys.stdout
        sys.stdout = nullwrite # disable kmeansoutput

        p_label, p_acc, p_val = svm_predict([0], instance, self.models[model_index], '-b 1')

        sys.stdout = oldstdout
        return p_label, p_acc, p_val

    def getHeterogenousClusterId(self):
        """
        Gets clusters ids which contain instances of different class, therefore heterogenous
        :return: feature ids of cluster heterogenous clusters, list of cluster ids
        """
        heterogenousClusters, heterogenousClusterId  = [], []
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

    def getSupportVectors(self):
        self.supportVectors = []
        for model in self.models.values():
            self.supportVectors.extend(model.get_SV())

        return self.supportVectors

