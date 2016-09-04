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
import random
from extractor import *
from FileIO import *
from trainer import *
import numpy as np
import copy


class ParallelTrainer:
    """Trainer Class used for the parallel training"""

    def __init__(self, name, files, klist, getTrainingDataFrom ='../data/trainingData/', trainingAdress = '../data/newMethodTraining/'):

        def assertIt(files,klist):
            """The function that checks parameters"""
            if(len(files)!=len(klist)):
                raise NameError("K list array is not convenient with the groups")

            for i in range(len(files)):
                if len(files[i])>klist[i]:
                    raise NameError("K list array is not convenient with the groups")

        assertIt(files,klist)
        #Controls if we should print
        self.debugMode = True

        self.name = name

        #Use the following data folder
        self.getTrainingDataFrom = getTrainingDataFrom

        #No of classes in a group
        self.numOfGroups = len(files)

        #Training address for saving and loading
        self.trainingAdress = trainingAdress

        self.klist = klist
        self.files = files


        if self.debugMode:
            print 'Our files are ', self.files

    def getFeatures(self, i, numclass, numfull, numpartial, trainingDataFolder):
        """The function that imports features

        !Assumes in training folder there is a csv folder and under it there are folders with names of classes
        also assumes under every class folder there is a csv file with the same name!
        """
        if(self.debugMode):
            print "GETTING FEATURES IN THE PARALLEL TRAINER"
        #Extract data for the ith group
        extr = Extractor(trainingDataFolder)
        features, isFull, classId, names, folderList = extr.loadFoldersParallel(numclass   = numclass,
                                                                                numfull    = numfull,
                                                                                numpartial = numpartial,
                                                                                folderList = self.files[i])
        if(self.debugMode):
            print "YEP GOT THEM"


        return features,isFull,classId,names,folderList


    def trainSVM(self, numclass, numfull, numpartial):
        """The trainer function"""

        fio = FileIO()
        normalProb = []

        import os
        path = self.trainingAdress + self.name

        #Check if the save directory exists
        if not os.path.exists(path):
            os.mkdir(path)

        import imp
        try:
            imp.find_module('pycuda')
            found = True
        except ImportError:
            found = False

        #For every group
        for i in range(len(self.files)):
            #Get k for this group
            k = self.klist[i]

            #Saving name for the specific group
            trainingName = '%s_%i__CFPK_%i_%i_%i_%i' % ('training',i, numclass, numfull, numpartial, k)
            trainingpath = path +'/'+ trainingName

            #Get features for this group
            features, isFull, classId, names, folderlist = self.getFeatures(i,numclass,numfull,numpartial,self.getTrainingDataFrom)


            #Get clusters for this group
            if not found:
                if(self.debugMode):
                    print "--------Training is Done With NORMAL CKMEANS--------"

                from complexCKMeans import *
                ckmeans = ComplexCKMeans(features, isFull, classId, k = k)
                kmeansoutput = ckmeans.getCKMeans()
            else:
                if(self.debugMode):
                    print "--------Training is Done With CUDA--------"

                from complexcudackmeans import *
                clusterer = complexCudaCKMeans(features, k, classId, isFull)  # FEATURES : N x 720
                clusters, centers = clusterer.cukmeans()
                kmeansoutput = [clusters, centers]

            #Train svm in this group

            trainer = Trainer(kmeansoutput, classId, features)
            heteClstrFeatureId, heteClstrId = trainer.getHeterogenousClusterId()
            trainer.trainSVM(heteClstrFeatureId, trainingpath)
            fio.saveTraining(names, classId, isFull, features, kmeansoutput,
                             trainingpath, trainingName)


            #NOW CALCULATE THE CENTER OF THE GROUP
            nowCenter = np.zeros(len(features[0]))
            totalNumOfInstances = len(features)
            for cluster in kmeansoutput[0]:
                for instance in cluster:
                    nowCenter += features[instance]
            nowCenter = nowCenter/totalNumOfInstances

            #SAVE THE CENTER
            fio.saveOneFeature(trainingpath +'/' + str(i) + "__Training_Center_",nowCenter)

            #Add total num of instances for calculating normalprob
            normalProb.append(totalNumOfInstances)

        #NOW CREATE Normal PROB
        kj = 0
        for i in normalProb:
            kj += i
        for i in range(len(normalProb)):
            normalProb[i] = float(normalProb[i])/kj


        #Save some important stuff  (MOSTLY FOR PASSING PRIORPROB to the predictor
        trainingInfo = {'normalProb':normalProb, 'numclass':numclass, 'numfull':numfull, 'numpartial':numpartial, 'numTrain':len(self.files),
                        'files':self.files, 'klist':self.klist}
        np.save(path+'/trainingInfo.npy', trainingInfo)




