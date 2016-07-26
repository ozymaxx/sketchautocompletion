
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

def main():
    features = list()
    classId = list()
    isFull = list()
    #pathdir = os.listdir(path)
    pathdir = ['airplane', 'alarm-clock', 'angel', 'ant', 'apple']
    classIdCount = 0

    path = '../json/'
    NUMFULLSKETCHPERCLASS = 2
    NUMPARTIALSKETCHPERFULL = 12
    NUMCLASS = 2
    classIdCount= 0
    for folder in pathdir:
        # iterate over folders in pathdir
        if classIdCount == NUMCLASS:
            break
        folderdir = os.listdir(path + folder)
        sketchcounter = 0

        for sketchcounter in range(1, NUMFULLSKETCHPERCLASS):
            # iterate over sketches

            fullsketchpath = path + folder + '/' + folder + '_' + str(sketchcounter) + '.json'
            if not os.path.isfile(fullsketchpath):
                break

            print fullsketchpath
            feature = featureExtract(fullsketchpath)
            features.append(np.array(feature))
            classId.append(classIdCount)
            isFull.append(1)
            partialsketchcount = 1
            while os.path.isfile(path + folder + '/' + folder + '_' + str(sketchcounter) + "_" + str(partialsketchcount) + '.json'):
                if partialsketchcount == NUMPARTIALSKETCHPERFULL:
                    break
                print path + folder + '/' + folder + '_' + str(sketchcounter) + "_" + str(partialsketchcount) + '.json'
                feature = featureExtract(path + folder + '/' + folder + '_' + str(sketchcounter) + "_" + str(partialsketchcount) + '.json')
                features.append(np.array(feature))
                classId.append(classIdCount)
                isFull.append(0)
                partialsketchcount += 1
        classIdCount += 1

    files = os.listdir('../classifiers/')
    for file in files:
        extension = os.path.splitext(file)[1]
        if extension == '.model':
			os.remove('../classifiers/'+file)

	NUMPOINTS = len(features)
	test = getConstraints(NUMPOINTS, isFull, classId)
	ckmeans = CKMeans(test, np.transpose(features), NUMCLASS)
	kmeansoutput = ckmeans.getCKMeans()

	# find heterogenous clusters and train svm
	trainer = Trainer(kmeansoutput, classId, features) ### FEATURES : TRANSPOSE?
	heteClstrFeatureId, heteClstrId = trainer.getHeterogenous()
	trainer.trainSVM(heteClstrFeatureId)
	# find the probability of given feature to belong any of the classes
	priorClusterProb = trainer.computeProb()
	predictor = Predictor(kmeansoutput,classId)
	outDict = predictor.calculateProb(featureExtract('../json/airplane/airplane_1.json'), priorClusterProb)

    print outDict


if __name__ == '__main__':
    main()
    #profile.run('print main(); print')
