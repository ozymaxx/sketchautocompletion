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


    myfio = FileIO()
    myfio.save(train_isFull, train_names, train_features, '../trainingData/csv/' + saveName+'/'+saveName+'.csv')
    myfio.save(test_isFull,test_names,test_features, '../testingData/csv/'+ saveName+'/'+saveName+'.csv')

    return train_features, train_isFull, train_classId, train_names, test_features, test_isFull, test_names, test_classId
#
#     '''
#     for j in range(numclass):
#         for i in range(numtestdata):
#             # take first partial sketch to be removed
#
#             # select test data randomly
#             # at most numtestdata for each class
#
#             index = 0
#             if selectTestRandom:
#                 index = np.random.randint(0, len(features))
#                 while test_classId.count(classId[index]) >= numtestdata:
#                     index = np.random.randint(0, len(features))
#             else:
#                 while test_classId.count(classId[index]) >= numtestdata:
#                     index += 1
#
#             classname, sketchid = processName(names[index])
#
#             # get test data mask
#             cond = [names[partialindex].split('_')[0] == classname and names[partialindex].split('_')[1] == sketchid for partialindex in range(len(features))]
#
#             # get only the full sketches for test data
#             test_features.extend([features[index] for index in range(len(features)) if cond[index]])
#             test_isFull.extend([isFull[index] for index in range(len(isFull)) if cond[index]])
#             test_names.extend([names[index] for index in range(len(names)) if cond[index]])
#             test_classId.extend([classId[index] for index in range(len(classId)) if cond[index]])
#
#             # remove any partial sketch from the traning data
#             features = [features[index] for index in range(len(features)) if not cond[index]]
#             names = [names[index] for index in range(len(names)) if not cond[index]]
#             isFull = [isFull[index] for index in range(len(isFull)) if not cond[index]]
#             classId = [classId[index] for index in range(len(classId)) if not cond[index]]
#
#     return train_features, train_isFull, train_classId, train_names, test_features, test_isFull, test_names, test_classId
#     '''
#
# def partitionfeatures_notin(features, isFull, classId, names, trainnames, numtestpartial, selectTestRandom = True):
#     numclass = len(set(classId))
#     print 'Starting partioning features'
#     cond = [bool(names[index] in trainnames) for index in range(len(features))]
#     test_features = [features[index] for index in range(len(features)) if not cond[index]]
#     test_isFull = [isFull[index] for index in range(len(isFull)) if not cond[index]]
#     test_classId = [classId[index] for index in range(len(classId)) if not cond[index]]
#     test_names = [names[index] for index in range(len(names)) if not cond[index]]
#
#     train_features = [features[index] for index in range(len(features)) if cond[index]]
#     train_isFull = [isFull[index] for index in range(len(isFull)) if cond[index]]
#     train_classId = [classId[index] for index in range(len(classId)) if cond[index]]
#     train_names = [names[index] for index in range(len(names)) if cond[index]]
#
#     '''
#     Divided data into train and test
#     But now we need to get numtestfull full sketch and
#     numtestpartial partial sketch
#     '''
#
#     test_full_features = [test_features[index] for index in range(len(test_features)) if test_isFull[index]]
#     test_full_classId = [test_classId[index] for index in range(len(test_classId)) if test_isFull[index]]
#     test_full_names = [test_names[index] for index in range(len(test_names))  if test_isFull[index]]
#     test_full_isFull = [test_isFull[index] for index in range(len(test_isFull)) if test_isFull[index]]
#
#     test_partial_features = [test_features[index] for index in range(len(test_features)) if not test_isFull[index]]
#     test_partial_classId = [test_classId[index] for index in range(len(test_classId)) if not test_isFull[index]]
#     test_partial_names = [test_names[index] for index in range(len(test_names))  if not test_isFull[index]]
#     test_partial_isFull = [test_isFull[index] for index in range(len(test_isFull)) if not test_isFull[index]]
#
#     from random import randint
#     while len(test_partial_features) > numtestpartial*numclass:
#         index = randint(0, len(test_partial_features)-1)
#         test_partial_features.pop(index)
#         test_partial_classId.pop(index)
#         test_partial_names.pop(index)
#         test_partial_isFull.pop(index)
#
#     '''
#     numtestfull = min(numtestfull, len(test_full_features))
#     numtestpartial = min(numtestpartial, len(test_partial_features))
#
#     if selectTestRandom:
#         import random
#         test_full_features = random.sample(test_full_features, numtestfull)
#         test_full_isFull = random.sample(test_full_isFull, numtestfull)
#         test_full_classId = random.sample(test_full_classId, numtestfull)
#         test_full_names = random.sample(test_full_names, numtestfull)
#
#         test_partial_features = random.sample(test_partial_features, numtestpartial)
#         test_partial_isFull = random.sample(test_partial_isFull, numtestpartial)
#         test_partial_classId = random.sample(test_partial_classId, numtestpartial)
#         test_partial_names = random.sample(test_partial_names, numtestpartial)
#     else:
#         test_full_features = test_full_features[:numtestfull]
#         test_full_isFull = test_full_isFull[:numtestfull]
#         test_full_classId = test_full_classId[:numtestfull]
#         test_full_names = test_full_names[:numtestfull]
#
#         test_partial_features = test_partial_features[:numtestpartial]
#         test_partial_isFull = test_partial_isFull[:numtestpartial]
#         test_partial_classId = test_partial_classId[:numtestpartial]
#         test_partial_names = test_partial_names[:numtestpartial]
#
#     if len(test_features) < numtestdata:
#         raise Warning('not enough test data')
#
#     if selectTestRandom:
#         import random
#         test_features = random.sample(test_features, numtestdata)
#         test_isFull = random.sample(test_isFull, numtestdata)
#         test_classId = random.sample(test_classId, numtestdata)
#         test_names = random.sample(test_names, numtestdata)
#     else:
#         test_features = test_features[:numtestdata]
#         test_isFull = test_isFull[:numtestdata]
#         test_classId = test_classId[:numtestdata]
#         test_names = test_names[:numtestdata]
#     '''
#
#
#     test_features = test_full_features + test_partial_features
#     test_isFull = test_full_isFull + test_partial_isFull
#     test_classId = test_full_classId + test_partial_classId
#     test_names = test_full_names + test_partial_names
#
#     print 'Partitioning End'
#     return train_features, train_isFull, train_classId, train_names, test_features, test_isFull, test_names, test_classId
#
# def xfoldpartition(
#     xfold,
#     whole_features,
#     whole_isFull,
#     whole_classId,
#     whole_names,
#     folderList):
#
#     '''
#        Assumes sketchids start from zero
#        and increment one by one, as extractor does
#     '''
#
#     xfold_features, xfold_names, xfold_classId, xfold_isFull = [], [], [], []
#
#     maxSketchId = 0
#     for name in whole_names:
#         _, SketchId, _ = processName(name)
#         if int(SketchId) > maxSketchId:
#             maxSketchId = int(SketchId)
#
#     for iter in range(xfold):
#         cond = [None]*len(whole_features)
#         for index in range(len(cond)):
#             _, sketchid, _ = processName(whole_names[index])
#             cond[index] = bool((iter*1.0/xfold)*maxSketchId < sketchid) and \
#                           bool(sketchid <= ((iter+1)*1.0/xfold)*maxSketchId)
#
#         xfold_features.append([whole_features[index] for index in range(len(whole_features)) if cond[index]])
#         xfold_isFull.append([whole_isFull[index] for index in range(len(whole_isFull)) if cond[index]])
#         xfold_classId.append([whole_classId[index] for index in range(len(whole_classId)) if cond[index]])
#         xfold_names.append([whole_names[index] for index in range(len(whole_names)) if cond[index]])
#
#     if sum([len(f) for f in xfold_features]) != len(whole_features):
#         raise Exception
#
#     return xfold_features, xfold_isFull, xfold_classId, xfold_names