"""
Ahmet BAGLAN
"""

import sys
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../data/')
from extractor import *
from FileIO import *
import numpy as np


def saveCenters(doOnlyFulls = False, trainingFolder = '../data/nicicon/csv/train/csv/' , suffix = '_train.csv', savePath = '../data/newMethodTraining/allCentersNic.csv'):
    """This file computes representatives of classes in the nicicon database and saves it to  the ./data/csv/allCentersNic.csv"""

    files = ['accident','bomb','car','casualty','electricity','fire','firebrigade','flood','gas','injury','paramedics', 'person', 'police','roadblock']
    centerSet  = []
    my_names = []
    my_isFull = []
    f = FileIO()


    extr_train = Extractor(trainingFolder)
    train_features, \
    train_isFull, \
    train_classId, \
    train_names = extr_train.loadniciconfolders()

    for mfile in files[:]:
        #Get every instance in the given class
        names, isFull, features = f.load(trainingFolder + mfile + '/' + mfile + suffix)

        #Initialise centers as 0
        nowCenter = np.zeros(len(features[0]))

        if not doOnlyFulls:
            #Use every full and partial instance in the class and sum them
            totalNumOfInstances = len(features)
            for instance in features:
                nowCenter += instance
        else:
            #Use only full instances in the class and sum them
            totalNumOfInstances = 0
            for i in range(len(features)):
                if isFull[i] == True:#Check the prints to see if this works fine (See if the totalNumOfInstances is meaningful
                    nowCenter+=features[i]
                    totalNumOfInstances+=1

        print mfile,'Used ', totalNumOfInstances, 'of instances to find the class center'

        nowCenter = nowCenter/totalNumOfInstances

        centerSet.append(nowCenter)#Add the mean feature vector for this representative
        my_names.append(mfile)#The name of the representative
        my_isFull.append(0)#Just set anything, not necessary for class representative
    f.save(my_isFull,my_names,centerSet,savePath)


if __name__ == '__main__':saveCenters(True, suffix='.csv')