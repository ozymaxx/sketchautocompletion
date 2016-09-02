"""
Predictor
Ahmet BAGLAN
"""


import sys
sys.path.append('../classifiers')
sys.path.append("../../libsvm-3.21/python/")
sys.path.append('../data/')
from FileIO import *
from trainer import *
from shapecreator import *
from FeatureExtractor import *
from ParallelPredictorSlave import *
from scipy.spatial import distance

import operator

class ParallelPredictorMaster:
    """The predictor class implementing functions to return probabilities"""
    def __init__(self, name):

        self.debugMode = False

        #The path of the saved training
        #!!newMethodTraining is static
        path = trainingpath = '../data/newMethodTraining/' + name

        #Load info of the saving
        trainInfo = np.load(path + '/trainingInfo.npy').item()


        self.groupCenters = []
        self.name = name
        #Get priorProb , num of groups in the training , numfull, numpartial, numclass, k in the training
        self.normalProb = trainInfo['normalProb']
        self.numOfTrains = trainInfo['numTrain']
        numclass = trainInfo['numclass']
        numfull = trainInfo['numfull']
        numpartial = trainInfo['numpartial']
        k = trainInfo['k']

        #Slave predictors
        self.predictors = []

        #Load slave predictors
        fio = FileIO()
        for i in range(self.numOfTrains):
            trainingName = '%s_%i__CFPK_%i_%i_%i_%i' % ('training',i, numclass, numfull, numpartial, k)
            trainingpath = '../data/newMethodTraining/' + name+ '/' + trainingName

            #get the center of the group
            b = fio.loadOneFeature(trainingpath +'/' + str(i) + "__Training_Center_")
            self.groupCenters.append(b)

            #get the slave predictor of the group
            names, classId, isFull, features, kmeansoutput, loadedFolders = fio.loadTraining(trainingpath + "/" + trainingName)
            loadedFolders = trainInfo['files'][i]
            nowSwm = Trainer.loadSvm(kmeansoutput, classId, trainingpath, features)
            if self.debugMode:
                print '+++++++++++++++++++++++++++++++++++++++++++++++++'
                print trainInfo
                print '+++++++++++++++++++++++++++++++++++++++++++++++++'

            #WE Give the file names to the slave predictors
            #Their output is directly by name s.a {'airplane:0.1}
            predictor = ParallelPredictorSlave(kmeansoutput, classId, trainingpath, loadedFolders, nowSwm)
            self.predictors.append(predictor)


    def probToTheGroupByFeature(self, instance, centers, priorProb):
        """Get prob to groups when instance is given as vector"""

        dist = [self.getDistance(instance, centers[idx]) for idx in range(len(centers))]

        sigma = 0.3
        mindist = min(dist)
        diste_ = [math.exp(-1*abs(d - mindist)/(2*sigma*sigma)) for d in dist]
        clustProb = [diste_[idx]*priorProb[idx] for idx in range(len(centers))]

        # normalize
        clustProb = [c/sum(clustProb) for c in clustProb]

        return clustProb

    def getDistance(self, x, y):
        """Computes euclidian distance between x instance and y instance
        inputs: x,y instances
        kmeansoutput: distance"""
        return distance.euclidean(x, y)

    def predictByString(self, jstring):

        loadedSketch = shapecreator.buildSketch('json', jstring)
        featextractor = IDMFeatureExtractor()
        instance = featextractor.extract(loadedSketch)

        classProb = self.calculatePosteriorProb(instance)
        return classProb

    def calculatePosteriorProb(self, ins):

        normalProb = self.normalProb
        l = self.groupCenters
        #The probability to being in some group
        savingProbs = self.probToTheGroupByFeature(ins, l, normalProb)
        savingProbs = [x/sum(savingProbs) for x in savingProbs]
        if(self.debugMode):
            print savingProbs

        out = dict()
        #Foreach slave predictor
        for i in range(self.numOfTrains):
            #Let the slave guess
            a = self.predictors[i].predictIt(ins)
            if self.debugMode:
                print "slave predicted ",sorted(a.items(), key=operator.itemgetter(1))
            for m in a.keys():
                out[m] = a[m]*savingProbs[i]

        return out





