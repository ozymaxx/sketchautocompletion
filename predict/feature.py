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

from FeatureExtractor import *
from shapecreator import *

import numpy as np
from svmutil import *
import matplotlib.pyplot as plt
from ckmeans import *
from getConstraints import *
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
	for i in range(1,79):
		filename = '../json/airplane/airplane_' + str(i) + ".json"
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

	NUMPOINTS = len(features)
	isFull = [0]*NUMPOINTS

	test = getConstraints(NUMPOINTS, isFull, classId)
	kmeans = CKMeans(test,np.transpose(features),2)
	output = kmeans.getCKMeans()
	a = 5

    # print computeProb(output)
#################################################
    # for all clusters
	'''
    clustersToBeTrained, toBeTrainedId = getHeterogenous(output,classId)
    allSV = trainSVM(np.transpose(features), clustersToBeTrained, classId)

    visualise.visualiseAfterClustering(allSV,output,np.transpose(features), classId, isFull, centers, "cluster number not defined")
    plt.show()
	'''
if __name__ == '__main__':
    main()
    print "fin"
    #profile.run('print main(); print')