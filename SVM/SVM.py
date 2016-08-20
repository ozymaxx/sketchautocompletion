import sys
sys.path.append('../classifiers')
sys.path.append("../../libsvm-3.21/python")
sys.path.append('../data/')
from svmutil import *
import copy_reg
import types
import copy

class SVM:
    def __init__(self, kmeansoutput, classId, subDirectory, features):
        self.kmeansoutput = kmeansoutput
        self.classId = classId
        self.subDirectory = subDirectory
        self.models = {}
        self.features = features
        self.allSV = []
        
    def multi_run_wrapper(self, args):
        return self.trainSVM2(*args)
    
    def _pickle_method(self, m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)
    
    def doSVM(self, clusterIdArr, directory):
        copy_reg.pickle(types.MethodType, self._pickle_method)
        from multiprocessing import Pool
        import itertools
        pool = Pool(4)
        pool.map(self.multi_run_wrapper,[(clusterIdArr,directory,0),(clusterIdArr,directory,1),
            (clusterIdArr,directory,2),(clusterIdArr,directory,3)])
    
    def trainSVM(self, clusterIdArr, directory):
        """Trains the support vector machine and saves models
        Inputs : clusterIdArr: list of cluster id's to be trained (heto)
        Outputs : Support VectorS
        """
        print 'Training SVM with %i features' % sum(len(cluster) for cluster in clusterIdArr)
        #label = copy.copy(self.classId) WHY
        label = self.classId
        order = 0
        for clusterFeatureId in clusterIdArr:  # Foreach cluster
            y = []
            x = []
            for featureid in clusterFeatureId:  # Foreach instance in the cluster
                featureid = int(featureid)
                y.append(label[featureid])
                x.append(self.features[featureid].tolist())

            problem = svm_problem(y, x)
            param = svm_parameter('-s 0 -t 2 -g 0.125 -c 8 -b 1 -q')
            
            m = svm_train(problem, param)
            self.allSV.extend(m.get_SV())

            import os
            if directory and not os.path.exists(directory):
                os.mkdir(directory)
            
            if directory:
                svm_save_model(directory + "/" + "clus" + str(order) + '.model', m)  # Save the model for the cluster
                print 'Saved Model %s' % str(directory + "/" + "clus" + str(order) + '.model')
            self.models[order] = m
            order += 1
        print 'Training SVM is done'
        
    def trainSVM2(self, clusterIdArr, directory, procId):
        """Trains the support vector machine and saves models
        Inputs : clusterIdArr: list of cluster id's to be trained (heto)
        Outputs : Support VectorS
        """
        print 'Training SVM with %i features' % sum(len(cluster) for cluster in clusterIdArr)

        #label = copy.copy(self.classId) WHY
        label = self.classId
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

            print "start train"
            m = svm_train(problem, param)
            print "finish train"
            self.allSV.extend(m.get_SV())

            import os
            if directory and not os.path.exists(directory):
                os.mkdir(directory)
            
            if directory:
                svm_save_model(directory + "/" + "clus" + str(order) + '.model', m)  # Save the model for the cluster
                print 'Saved Model %s' % str(directory + "/" + "clus" + str(order) + '.model')
            self.models[order] = m
            order += 4

 
        print 'Training SVM is done'
    def getlabels(self, modIndex):
        return self.models[int(modIndex)].get_labels()

    def loadModels(self):
        import os
        order = 0
        while os.path.exists(self.subDirectory + "/" + "clus" + str(order) + '.model'):
            print 'Loading SVM Model %s ' % str(self.subDirectory + "/" + "clus" + str(order) + '.model')
            self.models[order] = svm_load_model(self.subDirectory + "/" + "clus" + str(order) + '.model')
            order += 1

    def predict(self, modIndex, instance):
        if not self.models:
            order = 0
            import os
            while os.path.exists(self.subDirectory + "/" + "clus" + str(order) + '.model'):
                self.models[order] = svm_load_model(self.subDirectory + "/" + "clus" + str(order) + '.model')
                order += 1

        import sys
        class NullWriter(object):
            def write(self, arg):
                pass
        nullwrite = NullWriter()
        oldstdout = sys.stdout
        sys.stdout = nullwrite # disable kmeansoutput

        p_label, p_acc, p_val = svm_predict([0], instance, self.models[modIndex], '-b 1')

        sys.stdout = oldstdout
        return p_label, p_acc, p_val

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

    def getSV(self):
        self.allSV = []
        for model in self.models.values():
            self.allSV.extend(model.get_SV())

        return self.allSV

