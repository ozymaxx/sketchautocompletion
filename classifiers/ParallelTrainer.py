import sys
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append('../test/')
sys.path.append('../data/')
sys.path.append("../../libsvm-3.21/python")
import copy
import random
from extractor import *
from FileIO import *
import numpy as np

class ParallelTrainer:
    """Trainer Class used for the training"""

    def __init__(self, n, files, doKMeans = True, myAllTrainingData = None):
        self.getFromCSV = True
        if(myAllTrainingData!=None):
            self.myAllTrainingData = myAllTrainingData
            self.getFromCSV=False

        self.n = n
        self.trainingAdress = '../data/newMethodTraining/'
        allFiles = copy.copy(files)
        self.files = []

        if(not doKMeans):
            random.shuffle(allFiles)
            for i in range(len(files)/self.n):
                self.files.append(allFiles[i*n :(i+1)*n])

        else:
            print "NOW DOING NORMAL KMEANS FOR CLUSTERING CLASSES"
            f = FileIO()
            names, isFull,my_features = f.load('../data/csv/allCenters.csv')
            features = []
            for n in files:
                features.append(my_features[names.index(n)])
            features = np.array(features)

            classId = range(len(features))
            constarr = getConstraints(size=len(features), isFull=isFull, classId=classId)
            ckmeans = CKMeans(constarr, np.transpose(features), len(files)/self.n)
            kmeansoutput = ckmeans.getCKMeans()
            for i in kmeansoutput[0]:
                l = []
                for j in i:
                    l.append(allFiles[j])
                self.files.append(l)

    def getFiles(self):
        return self.files

    def set(self,f):
        self.files = f

    def getFeatures(self, i, numclass, numfull,numpartial, normal = False):
        if(normal):
            extr = Extractor('../data/')
            features, isFull, classId, names, folderList = extr.loadfolders(  numclass   = numclass,
                                                                              numfull    = numfull,
                                                                              numpartial = numpartial,
                                                                       folderList = self.files[i])
        else:
            extr = Extractor('../trainingData/')
            features, isFull, classId, names, folderList = extr.loadfolders(  numclass   = numclass,
                                                                              numfull    = numfull,
                                                                              numpartial = numpartial,
                                                                       folderList = self.files[i])

        return features,isFull,classId,names,folderList


    def trainSWM(self, numclass, numfull, numpartial, k, name):
        n = self.n
        fio = FileIO()
        normalProb = []
        import os
        path = self.trainingAdress + name

        if not os.path.exists(path):
            os.mkdir(path)

        import imp
        try:
            imp.find_module('pycuda')
            found = True
        except ImportError:
            found = False


        for i in range(len(self.files)):
            trainingName = '%s_%i__CFPK_%i_%i_%i_%i' % ('training',i, numclass, numfull, numpartial, k)
            trainingpath = path +'/'+ trainingName
            features, isFull, classId, names, folderList = self.getFeatures(i,numclass,numfull,numpartial,self.getFromCSV)

            if not found:
                constarr = getConstraints(size=len(features), isFull=isFull, classId=classId)
                ckmeans = CKMeans(constarr, np.transpose(features), k)
                kmeansoutput = ckmeans.getCKMeans()
            else:
                print "CUDDDDAAAAAAAAAAAAAAAAAAAAAAAA"
                from cudackmeans import *
                clusterer = CuCKMeans(features, k, classId, isFull)  # FEATURES : N x 720
                clusters, centers = clusterer.cukmeans()
                kmeansoutput = [clusters, centers]

            # find heterogenous clusters and train svm
            trainer = Trainer(kmeansoutput, classId, features)
            heteClstrFeatureId, heteClstrId = trainer.getHeterogenous()
            trainer.trainSVM(heteClstrFeatureId, trainingpath)
            fio.saveTraining(names, classId, isFull, features, kmeansoutput,
                             trainingpath, trainingName)
            trainer.trainSVM(heteClstrFeatureId, trainingpath)

            nowCenter = np.zeros(len(features[0]))
            totalNumOfInstances = len(features)
            for cluster in kmeansoutput[0]:
                for instance in cluster:
                    nowCenter += features[instance]
            nowCenter = nowCenter/totalNumOfInstances
            fio.saveOneFeature(trainingpath +'/' + str(i) + "__Training_Center_",nowCenter)
            normalProb.append(totalNumOfInstances)
        kj = 0
        for i in normalProb:
            kj += i
        for i in range(len(normalProb)):
            normalProb[i] = float(normalProb[i])/kj


        trainingInfo = {'normalProb':normalProb, 'k':k, 'numclass':numclass, 'numfull':numfull, 'numpartial':numpartial, 'numTrain':len(self.files)}
        np.save(path+'/trainingInfo.npy', trainingInfo)




