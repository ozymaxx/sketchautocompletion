"""
Predictor
Ahmet BAGLAN
14.07.2016
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

class ParallelPredictorMaster:
    """The predictor class implementing functions to return probabilities"""
    def __init__(self, name):


        path = trainingpath = '../data/newMethodTraining/' + name
        trainInfo = np.load(path + '/trainingInfo.npy').item()
        self.l = []
        self.name = name
        self.normalProb = trainInfo['normalProb']
        self.numOfTrains = trainInfo['numTrain']

        self.predictors = []
        numclass = trainInfo['numclass']
        numfull = trainInfo['numfull']
        numpartial = trainInfo['numpartial']
        k = trainInfo['k']

        fio = FileIO()
        for i in range(self.numOfTrains):
            trainingName = '%s_%i__CFPK_%i_%i_%i_%i' % ('training',i, numclass, numfull, numpartial, k)
            trainingpath = '../data/newMethodTraining/' + name+ '/' + trainingName
            b = fio.loadOneFeature(trainingpath +'/' + str(i) + "__Training_Center_")
            self.l.append(b)
            names, classId, isFull, features, kmeansoutput, loadedFolders = fio.loadTraining(trainingpath + "/" + trainingName)
            predictor = ParallelPredictorSlave(kmeansoutput, classId, trainingpath, loadedFolders)
            self.predictors.append(predictor)

    def probToSavings(self, jstring,  centers, normalProb):
        loadedSketch = shapecreator.buildSketch('json', jstring)
        featextractor = IDMFeatureExtractor()
        instance = featextractor.extract(loadedSketch)
        probTup = []
        for i in range(len(centers)):
            dist = self.getDistance(instance, centers[i])
            probTup.append(math.exp(-1*abs(dist))*normalProb[i])
        return probTup

    def getDistance(self, x, y):
        """Computes euclidian distance between x instance and y instance
        inputs: x,y instances
        kmeansoutput: distance"""
        x = np.asarray(x)
        y = np.asarray(y)
        return np.sqrt(np.sum((x-y)**2))

    def predictByString(self, q):

        normalProb = self.normalProb
        l = self.l
        savingProbs = self.probToSavings(q, l, normalProb)
        savingProbs = [x/sum(savingProbs) for x in savingProbs]
        out = dict()
        for i in range(self.numOfTrains):
            a = self.predictors[i].predictByString(q)
            for m in a.keys():
                out[m] = a[m]*savingProbs[i]
        return out





