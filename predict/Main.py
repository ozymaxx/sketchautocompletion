
import sys
import os
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append("../../libsvm-3.21/python/")

print sys.path
from trainer import *
from ckmeans import *
from getConstraints import *
from Predictor import *
from FeatureExtractor import *
from shapecreator import *
import numpy as np
from svmutil import *
from feature import *

class Main:
    def __init__(self):
        self.features = list()
        self.classId = list()
        self.isFull = list()
        self.path = '../json/'

    def doIt(self, NUMFULLSKETCHPERCLASS,NUMPARTIALSKETCHPERFULL,NUMCLASS):
        classIdCount= 0
        pathdir = ['airplane', 'alarm-clock', 'angel', 'ant', 'apple']
        for folder in pathdir:
            # iterate over folders in pathdir
            if classIdCount == NUMCLASS:
                break
            folderdir = os.listdir(self.path + folder)
            sketchcounter = 0

            for sketchcounter in range(1, NUMFULLSKETCHPERCLASS):
                # iterate over sketches

                fullsketchpath = self.path + folder + '/' + folder + '_' + str(sketchcounter) + '.json'
                if not os.path.isfile(fullsketchpath):
                    break

                print fullsketchpath
                feature = featureExtract(fullsketchpath)
                self.features.append(np.array(feature))
                self.classId.append(classIdCount)
                self.isFull.append(1)
                partialsketchcount = 1
                while os.path.isfile(self.path + folder + '/' + folder + '_' + str(sketchcounter) + "_" + str(partialsketchcount) + '.json'):
                    if partialsketchcount == NUMPARTIALSKETCHPERFULL:
                        break
                    print self.path + folder + '/' + folder + '_' + str(sketchcounter) + "_" + str(partialsketchcount) + '.json'
                    feature = featureExtract(self.path + folder + '/' + folder + '_' + str(sketchcounter) + "_" + str(partialsketchcount) + '.json')
                    self.features.append(np.array(feature))
                    self.classId.append(classIdCount)
                    self.isFull.append(0)
                    partialsketchcount += 1
            classIdCount += 1

        files = os.listdir('../classifiers/')
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == '.model':
                os.remove('../classifiers/'+file)

        NUMPOINTS = len(self.features)
        test = getConstraints(NUMPOINTS, self.isFull, self.classId)
        ckmeans = CKMeans(test, np.transpose(self.features), NUMCLASS)
        kmeansoutput = ckmeans.getCKMeans()

        # find heterogenous clusters and train svm
        trainer = Trainer(kmeansoutput, self.classId, self.features) ### FEATURES : TRANSPOSE?
        heteClstrFeatureId, heteClstrId = trainer.getHeterogenous()
        trainer.trainSVM(heteClstrFeatureId)
        # find the probability of given feature to belong any of the classes
        priorClusterProb = trainer.computeProb()
        predictor = Predictor(kmeansoutput,self.classId)
        outDict = predictor.calculateProb(featureExtract('../json/airplane/airplane_1.json'), priorClusterProb)

        print outDict



def main():

    m = Main()
    m.doIt(2,12,2)

if __name__ == '__main__':
    main()
    #profile.run('print main(); print')



