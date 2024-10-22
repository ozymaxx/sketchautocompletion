"""
Methods for partitioning features into test and training data
"""
import numpy as np

import sys
sys.path.append('../data/')
from FileIO import *


def processName(name):
    name_split = name.split('_')
    classname = name_split[0]
    sketchid = name_split[1]
    partialid = -1 if bool(len(name_split)==2) else name_split[2]

    return classname, int(sketchid), int(partialid)

'''
 Always select training data deterministically,
 Select test data according to selectTestRandom
 for easier reproduction of the results
'''

def partitionfeatures(features, isFull, classId, names, numtrainfull, selectTestRandom = True, saveName = None):

    numclass = len(set(classId))
    #test_features, test_names, test_classId, test_isFull = [], [], [], []

    # Find the
    '''
    sketchPartialMax = dict()
    for name in names:
        classname, sketchid, partialid = processName(name)
        sketchPartialMax[classname + '_' + sketchid] = max(sketchPartialMax[classname + '_' + sketchid], partialid)
    '''
    # condition for being training data
    cond = [bool(processName(names[index])[1] < numtrainfull) for index in range(len(features))]

    test_features = [features[index] for index in range(len(features)) if not cond[index]]
    test_isFull = [isFull[index] for index in range(len(isFull)) if not cond[index]]
    test_names = [names[index] for index in range(len(names)) if not cond[index]]
    test_classId = [classId[index] for index in range(len(classId)) if not cond[index]]

    # remove any partial sketch from the traning data
    train_features = [features[index] for index in range(len(features)) if cond[index]]
    train_names = [names[index] for index in range(len(names)) if cond[index]]
    train_isFull = [isFull[index] for index in range(len(isFull)) if cond[index]]
    train_classId = [classId[index] for index in range(len(classId)) if cond[index]]


    # myfio = FileIO()
    # myfio.save(train_isFull, train_names, train_features, '../trainingData/' + saveName+'/'+saveName+'.csv')
    # myfio.save(test_isFull,test_names,test_features, '../testingData/'+ saveName+'/'+saveName+'.csv')

    return train_features, train_isFull, train_classId, train_names, test_features, test_isFull, test_names, test_classId

    '''
    for j in range(numclass):
        for i in range(numtestdata):
            # take first partial sketch to be removed

            # select test data randomly
            # at most numtestdata for each class

            index = 0
            if selectTestRandom:
                index = np.random.randint(0, len(features))
                while test_classId.count(classId[index]) >= numtestdata:
                    index = np.random.randint(0, len(features))
            else:
                while test_classId.count(classId[index]) >= numtestdata:
                    index += 1

            classname, sketchid = processName(names[index])

            # get test data mask
            cond = [names[partialindex].split('_')[0] == classname and names[partialindex].split('_')[1] == sketchid for partialindex in range(len(features))]

            # get only the full sketches for test data
            test_features.extend([features[index] for index in range(len(features)) if cond[index]])
            test_isFull.extend([isFull[index] for index in range(len(isFull)) if cond[index]])
            test_names.extend([names[index] for index in range(len(names)) if cond[index]])
            test_classId.extend([classId[index] for index in range(len(classId)) if cond[index]])

            # remove any partial sketch from the traning data
            features = [features[index] for index in range(len(features)) if not cond[index]]
            names = [names[index] for index in range(len(names)) if not cond[index]]
            isFull = [isFull[index] for index in range(len(isFull)) if not cond[index]]
            classId = [classId[index] for index in range(len(classId)) if not cond[index]]

    return train_features, train_isFull, train_classId, train_names, test_features, test_isFull, test_names, test_classId
    '''

def partitionfeatures_notin(features, isFull, classId, names, trainnames, numtest):
    numclass = len(set(classId))
    print 'Starting partioning features'
    cond = [bool(processName(names[index])[1] < 80-numtest) for index in range(len(features))]
    train_features = [features[index] for index in range(len(features)) if cond[index]]
    train_isFull = [isFull[index] for index in range(len(isFull)) if cond[index]]
    train_classId = [classId[index] for index in range(len(classId)) if cond[index]]
    train_names = [names[index] for index in range(len(names)) if cond[index]]

    test_features = [features[index] for index in range(len(features)) if not cond[index]]
    test_isFull = [isFull[index] for index in range(len(isFull)) if not cond[index]]
    test_classId = [classId[index] for index in range(len(classId)) if not cond[index]]
    test_names = [names[index] for index in range(len(names)) if not cond[index]]

    '''
    Divided data into train and test
    But now we need to get numtestfull full sketch and
    numtestpartial partial sketch
    '''
    print 'Partitioning End'
    return train_features, train_isFull, train_classId, train_names, test_features, test_isFull, test_names, test_classId

def xfoldpartition(
    xfold,
    whole_features,
    whole_isFull,
    whole_classId,
    whole_names,
    folderList):

    '''
       Assumes sketchids start from zero
       and increment one by one, as extractor does
    '''

    xfold_features, xfold_names, xfold_classId, xfold_isFull = [], [], [], []

    maxSketchId = 0
    for name in whole_names:
        _, SketchId, _ = processName(name)
        if int(SketchId) > maxSketchId:
            maxSketchId = int(SketchId)

    for iter in range(xfold):
        cond = [None]*len(whole_features)
        for index in range(len(cond)):
            _, sketchid, _ = processName(whole_names[index])
            cond[index] = bool((iter*1.0/xfold)*maxSketchId < sketchid) and \
                          bool(sketchid <= ((iter+1)*1.0/xfold)*maxSketchId)

        xfold_features.append([whole_features[index] for index in range(len(whole_features)) if cond[index]])
        xfold_isFull.append([whole_isFull[index] for index in range(len(whole_isFull)) if cond[index]])
        xfold_classId.append([whole_classId[index] for index in range(len(whole_classId)) if cond[index]])
        xfold_names.append([whole_names[index] for index in range(len(whole_names)) if cond[index]])

    if sum([len(f) for f in xfold_features]) != len(whole_features):
        raise Exception

    return xfold_features, xfold_isFull, xfold_classId, xfold_names