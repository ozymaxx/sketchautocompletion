"""
Feature extraction
Arda ICMEZ
14.07.2016
"""

import sys
import os
sys.path.append("../../sketchfe/sketchfe")
import sys
sys.path.append('../predict/')
from feature import *

from FeatureExtractor import *
from shapecreator import *

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

feature = featureExtract('../json/castle/castle_3.json')
print feature