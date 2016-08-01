
import sys

sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append("../../libsvm-3.21/python/")

from trainer import *
from ckmeans import *
from getConstraints import *
from FeatureExtractor import *
from shapecreator import *
from feature import *

class Main:
    """The main class to be called"""
    def __init__(self):
        self.features = list()
        self.classId = list()
        self.isFull = list()
        self.path = '../json/'
        self.trainer = None
        self.kmeansoutput = None
        self.priorClusterProb = None
        self.d = {}
    def trainIt(self, NUMFULLSKETCHPERCLASS, NUMPARTIALSKETCHPERFULL, NUMCLASS, k):
        """P"""
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
                self.d[classIdCount] = folder
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
        ckmeans = CKMeans(test, np.transpose(self.features), k)
        self.kmeansoutput = ckmeans.getCKMeans()

        # find heterogenous clusters and train svm
        self.trainer = Trainer(self.kmeansoutput, self.classId, self.features) ### FEATURES : TRANSPOSE?
        heteClstrFeatureId, heteClstrId = self.trainer.getHeterogenous()
        self.trainer.trainSVM(heteClstrFeatureId)
        self.priorClusterProb = self.trainer.computePriorProb()


    def predictItInTheSet(self, instance):
        # find the probability of given feature to belong any of the classes
        loc = '../json/'+instance+'.json'
        predictor = Predictor(self.kmeansoutput,self.classId)
        outDict = predictor.calculatePosteriorProb(featureExtract(loc), self.priorClusterProb)
        d1= {}
        for i in outDict:
            d1[self.d[i]] = outDict[i]
        return d1

    def predictItUsingFeatures(self,feature):

        predictor = Predictor(self.kmeansoutput,self.classId)
        outDict = predictor.calculatePosteriorProb(feature, self.priorClusterProb)
        return outDict



def main():

    m = Main()
    m.trainIt(5, 12, 4,8)

    toBeChecked =[ 'airplane/airplane_80','airplane/airplane_78','airplane/airplane_77','ant/ant_15']

    for i in toBeChecked:
        print m.predictItInTheSet(i)
if __name__ == '__main__':
    main()
    #profile.run('print main(); print')



