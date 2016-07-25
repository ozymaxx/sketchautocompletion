"""
Feature extraction
Arda ICMEZ
14.07.2016
"""

import sys
import os
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append("../../libsvm-3.21/python/")
from feature import *
from trainer import *
from FeatureExtractor import *
from shapecreator import *

import numpy as np
from svmutil import *
import matplotlib.pyplot as plt
from ckmeans import *
from getConstraints import *
from clusterProb import *
import visualise

def featureExtract(filename):
    loadedSketch = None
    try:
        with open(filename,'rb') as f:
            extractType = None
            extension = os.path.splitext(filename)[1]

            if extension == '.json':
                extractType = 'json'
            elif extension == ".xml":
                extractType = "xml"

            elif extension == ".txt":
                extractType = "school"

            else:
                print "Wrong file type"
                exit

            filecontent = f.read()
            loadedSketch = shapecreator.buildSketch(extractType, filecontent)
    except:
        print "Error loading the sketch"
        exit

    featextractor = IDMFeatureExtractor()
    features = featextractor.extract(loadedSketch)
    return features

def main():
    features = list()
    classId = list()
    isFull = list()

    path = '../json/'
    #pathdir = os.listdir(path)
    pathdir = ['airplane', 'alarm-clock', 'angel', 'ant', 'apple']
    classIdCount = 0
    NUMFULLSKETCHPERCLASS = 2
    NUMPARTIALSKETCHPERFULL = 1
    NUMCLASS = 3
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
    heteClstrFeatureId, heteClstrId = getHeterogenous(kmeansoutput, classId)
    trainSVM(np.transpose(features), heteClstrFeatureId, classId)
    # find the probability of given feature to belong any of the classes
    priorClusterProb = computeProb(kmeansoutput)
    outDict = calculateProb(featureExtract('../json/airplane/airplane_1.json'), kmeansoutput, priorClusterProb, classId)

    print outDict
    a=5
    pass

    '''
	features = list()
	for i in range(1,79):
		filename = '../json/cigarette/cigarette_' + str(i) + ".json"
		print filename
		feature = featureExtract(filename)
		features.append(np.array(feature))
	#classId.extend([0]*len(features))


	NUMPOINTS = len(features)
	isFull = [0]*NUMPOINTS

	test = getConstraints(NUMPOINTS, isFull, classId)
	kmeans = CKMeans(test,np.transpose(features),2)
	output = kmeans.getCKMeans()
	a = 5
	'''
    '''
	for i in range(1,79):
		filename = '../json/cigarette/cigarette_' + str(i) + ".json"
		print filename
		feature = featureExtract(filename)
		features.append(np.array(feature))
	classId.extend([0]*len(features))

	for i in range(1,79):
		filename = '../json/castle/castle_' + str(i) + ".json"
		print filename
		feature = featureExtract(filename)
		features.append(np.array(feature))
	classId.extend([2]*len(features))
    # print computeProb(output)
#################################################
    # for all clusters
	'''
    '''
    clustersToBeTrained, toBeTrainedId = getHeterogenous(output,classId)
    allSV = trainSVM(np.transpose(features), clustersToBeTrained, classId)

    visualise.visualiseAfterClustering(allSV,output,np.transpose(features), classId, isFull, centers, "cluster number not defined")
    plt.show()
	'''
    '''
	feature = featureExtract('../json/alarm-clock_1_1.json')
	'''
if __name__ == '__main__':
    main()
    print "fin"
    #profile.run('print main(); print')