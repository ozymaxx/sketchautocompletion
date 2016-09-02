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
    """This file computes centers of classses and saves it to  the ./data/csv/allCenters.csv"""

    files = ['accident','bomb','car','casualty','electricity','fire','firebrigade','flood','gas','injury','paramedics', 'person', 'police','roadblock']
    k  = []
    my_names = []
    my_isFull = []
    f = FileIO()


    extr_train = Extractor(trainingFolder)
    train_features, \
    train_isFull, \
    train_classId, \
    train_names = extr_train.loadniciconfolders()

    for mfile in files[:]:

        names, isFull, features = f.load(trainingFolder + mfile + '/' + mfile + suffix)
        nowCenter = np.zeros(len(features[0]))

        if not doOnlyFulls:
            totalNumOfInstances = len(features)
            for instance in features:
                nowCenter += instance
        else:
            totalNumOfInstances = 0
            for i in range(len(features)):
                if isFull[i] == False:
                    nowCenter+=features[i]
                    totalNumOfInstances+=1
        print mfile, totalNumOfInstances
        nowCenter = nowCenter/totalNumOfInstances
        k.append(nowCenter)
        my_names.append(mfile)
        my_isFull.append(0)
    f.save(my_isFull,my_names,k,savePath)


if __name__ == '__main__':saveCenters(True, suffix='.csv')