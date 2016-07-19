"""
Feature extraction
Arda ICMEZ
14.07.2016
"""

import sys
import os
sys.path.append("../../sketchfe/sketchfe")

from sketchfe import shapecreator
import sketchfe.FeatureExtractor.IDMFeatureExtractor

def featureExtract(filename):
	loadedSketch = None
	
	try:
		with open(filename,'rb') as f:
			extractType = None
			extension = os.path.splitext(filename)[1]
			
			if extension == ".json":
				extractType = "json"
				
			elif extension == ".xml":
				extractType = "xml"
				
			elif extension == ".txt":
				extractType = "school"
				
			else:
				print "Wrong file type"
				exit
				
			filecontent = f.read()
			loadedSketch = shapecreator.buildSketch(filecontent,extractType)
	except:
		print "Error loading the sketch"
		exit
	
	featextractor = IDMFeatureExtractor()
	features = featextractor.extract(loadedSketch)
	
	