"""
Parallel trainer
Ahmet BAGLAN
"""

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
    """Trainer Class used for the parallel training"""

    def __init__(self, n, files, doKMeans = True, getTrainingDataFrom = '../data/trainingData/'):

        #Use the following data
        self.getTrainingDataFrom = getTrainingDataFrom
        #Controls if we should print
        self.debugMode = True
        #No of classes in a group
        self.n = n
        #Training address for saving and loading
        self.trainingAdress = '../data/newMethodTraining/'
        allFiles = copy.copy(files)
        self.files = []


        if(not doKMeans):#Create groups by Random Chosen
            random.shuffle(allFiles)
            for i in range(len(files)/self.n):
                self.files.append(allFiles[i*n :(i+1)*n])
        else:#Create groups using K-Means
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
        if self.debugMode:
            print 'Our files are ', self.files

    def getFeatures(self, i, numclass, numfull,numpartial, trainingDataFolder):

        if(self.debugMode):
            print "GETTING FEATURES IN THE PARALLEL TRAINER"
        extr = Extractor(trainingDataFolder)
        features, isFull, classId, names, folderList = extr.loadFoldersParallel(numclass   = numclass,
                                                                                numfull    = numfull,
                                                                                numpartial = numpartial,
                                                                                folderList = self.files[i])
        if(self.debugMode):
            print "YEP GOT THEM"

        return features,isFull,classId,names,folderList


    def trainSVM(self, numclass, numfull, numpartial, k, name):

        n = self.n
        fio = FileIO()
        normalProb = []

        import os
        path = self.trainingAdress + name

        #Check if the save directory exists
        if not os.path.exists(path):
            os.mkdir(path)

        import imp
        try:
            imp.find_module('pycuda')
            found = True
        except ImportError:
            found = False


        for i in range(len(self.files)):
            #Saving name for the specific group
            trainingName = '%s_%i__CFPK_%i_%i_%i_%i' % ('training',i, numclass, numfull, numpartial, k)
            trainingpath = path +'/'+ trainingName

            #Get features for the
            features, isFull, classId, names, folderlist = self.getFeatures(i,numclass,numfull,numpartial,self.getTrainingDataFrom)

            if(self.debugMode):
                print "Names------", names


            #NOW GET CLUSTERING OUTPUT FOR THE SPECIFIC GROUP
            if not found:
                if(self.debugMode):
                    print "--------Training is Done With NORMAL CKMEANS--------"

                constarr = getConstraints(size=len(features), isFull=isFull, classId=classId)
                ckmeans = CKMeans(constarr, np.transpose(features), k)
                kmeansoutput = ckmeans.getCKMeans()
            else:
                if(self.debugMode):
                    print "--------Training is Done With CUDA--------"

                from complexcudackmeans import *
                clusterer = complexCudaCKMeans(features, k, classId, isFull)  # FEATURES : N x 720
                clusters, centers = clusterer.cukmeans()
                kmeansoutput = [clusters, centers]

            #NOW TRAIN SVM FOR THE SPECIFIC GROUP
            # find heterogenous clusters and train svm

            trainer = Trainer(kmeansoutput, classId, features)
            heteClstrFeatureId, heteClstrId = trainer.getHeterogenous()
            trainer.trainSVM(heteClstrFeatureId, trainingpath)
            fio.saveTraining(names, classId, isFull, features, kmeansoutput,
                             trainingpath, trainingName)
            trainer.trainSVM(heteClstrFeatureId, trainingpath)

            #NOW CALCULATE THE CENTER OF THE GROUP
            nowCenter = np.zeros(len(features[0]))
            totalNumOfInstances = len(features)
            for cluster in kmeansoutput[0]:
                for instance in cluster:
                    nowCenter += features[instance]
            nowCenter = nowCenter/totalNumOfInstances

            #SAVE THE CENTER
            fio.saveOneFeature(trainingpath +'/' + str(i) + "__Training_Center_",nowCenter)
            normalProb.append(totalNumOfInstances)
        #NOW CREATE PRIOR PROB
        kj = 0
        for i in normalProb:
            kj += i
        for i in range(len(normalProb)):
            normalProb[i] = float(normalProb[i])/kj


        #Save some important stuff  (MOSTLY FOR PASSING PRIORPROB to the predictor
        trainingInfo = {'normalProb':normalProb, 'k':k, 'numclass':numclass, 'numfull':numfull, 'numpartial':numpartial, 'numTrain':len(self.files)}
        np.save(path+'/trainingInfo.npy', trainingInfo)




