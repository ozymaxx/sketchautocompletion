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

print sys.path
from ckmeans import *
from getConstraints import *
from Predictor import *
from FeatureExtractor import *
from shapecreator import *
from svmutil import *

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
