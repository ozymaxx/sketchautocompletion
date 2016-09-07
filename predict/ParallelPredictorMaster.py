"""
Predictor
Ahmet BAGLAN
"""


import sys
sys.path.append('../classifiers')
sys.path.append("../../libsvm-3.21/python/")
sys.path.append('../data/')
sys.path.append('../../sketchfe/sketchfe')
from FileIO import *
from trainer import *
from shapecreator import *
from FeatureExtractor import *
from ParallelPredictorSlave import *
from Predictor import *
from scipy.spatial import distance

import operator

class ParallelPredictorMaster:
    """The predictor class implementing functions to return probabilities for the alternative method"""

    def __init__(self, name, trainingFolder = '../data/newMethodTraining/'):
        #Controls prints
        self.debugMode = False

        #The path of training to be loaded
        path = trainingFolder+ name

        #Load info of the saving
        #trainingInfo.npy contains information about the groups such as which groups have which classes
        #what are prior probabilities (=numOfInstanceInTheGroup/totalNumOfInstances)
        trainInfo = np.load(path + '/trainingInfo.npy').item()

        self.groupCenters = []
        self.name = name

        #Get priorProb , num of groups in the training , numfull, numpartial, numclass, k in the training
        self.normalProb = trainInfo['normalProb']
        self.numOfTrains = trainInfo['numTrain']
        self.files = trainInfo['files']
        numclass = trainInfo['numclass']
        numfull = trainInfo['numfull']
        numpartial = trainInfo['numpartial']
        klist = trainInfo['klist']

        #Slave predictors
        self.predictors = []

        #Load slave predictors
        fio = FileIO()

        #Get prediction info in every group
        for i in range(self.numOfTrains):
            trainingName = '%s_%i__CFPK_%i_%i_%i_%i' % ('training',i, numclass, numfull, numpartial, klist[i])
            trainingpath = '../data/newMethodTraining/' + name+ '/' + trainingName

            #get the center of the group
            b = fio.loadOneFeature(trainingpath +'/' + str(i) + "__Training_Center_")
            self.groupCenters.append(b)

            #get the slave predictor of the group
            names, classId, isFull, features, kmeansoutput, loadedFolders = fio.loadTraining(trainingpath + "/" + trainingName)
            loadedFolders = trainInfo['files'][i]
            nowSwm = Trainer.loadSvm(kmeansoutput, classId, trainingpath, features)

            predictor = Predictor(kmeansoutput,classId,trainingpath,svm = nowSwm)
            self.predictors.append(predictor)


    def probToTheGroupByFeature(self, instance, groupCenter, priorProb):
        """Get prob to groups when instance is given as vector
        Same idea as with the clusterProb function
        """

        dist = [self.getDistance(instance, groupCenter[idx]) for idx in range(len(groupCenter))]

        sigma = 0.3
        mindist = min(dist)
        diste_ = [math.exp(-1*abs(d - mindist)/(2*sigma*sigma)) for d in dist]
        groupProb = [diste_[idx] * priorProb[idx] for idx in range(len(groupCenter))]

        # normalize
        groupProb = [c/sum(groupProb) for c in groupProb]

        return groupProb

    def getDistance(self, x, y):
        """Computes euclidian distance between x instance and y instance
        inputs: x,y instances
        kmeansoutput: distance"""
        return distance.euclidean(x, y)

    def predictByString(self, jstring):
        """Predictor function when input is jstring"""

        #Get features
        loadedSketch = shapecreator.buildSketch('json', jstring)
        featextractor = IDMFeatureExtractor()
        instance = featextractor.extract(loadedSketch)
        #Get output
        classProb = self.calculatePosteriorProb(instance)
        return classProb

    def getNormalDict(self,files, classProb):

        out = dict()
        for i in classProb:
            out[files[int(i)]]=classProb[i]
        return out

    def calculatePosteriorProb(self, ins):
        """Predictor function when input is jstring"""

        #The probability to being in some group
        groupProbs = self.probToTheGroupByFeature(ins, self.groupCenters, self.normalProb)
        groupProbs = [x/sum(groupProbs) for x in groupProbs]

        if(self.debugMode):
            print groupProbs

        out = dict()
        #Foreach slave predictor
        for i in range(self.numOfTrains):

            # #Let the slave guess
            # slaveProb = self.predictors[i].predictIt(ins)

            nowPredictor = self.predictors[i]
            priorClusterProb = nowPredictor.calculatePriorProb()
            classProbList = nowPredictor.calculatePosteriorProb(ins, priorClusterProb)
            slaveProb = self.getNormalDict(self.files[i], classProbList)

            if self.debugMode:
                print "slave predicted ",sorted(slaveProb.items(), key=operator.itemgetter(1))

            #slaveProb is the probabilities prediction in the group
            #groupProbs[i] is the probability to be in that group
            for m in slaveProb.keys():
                out[m] = slaveProb[m]*groupProbs[i]

        return out





