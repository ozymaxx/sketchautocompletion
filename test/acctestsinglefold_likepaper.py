"""
Script for accuracy test while splitting data non-overlapping test and
training parts. Results are saved into given directory as two pickle files
and selected graphs with selected parameters. Script saves two pickle files
as accuracy.p and reject.p which holds the accuracy and reject rates for
every possible k, N, C, isfull values respectively.

Having the pickle files, graphs can be redrawn (possibly with different parameters)
with acctestload script.
"""

import sys
import os
for path in os.listdir('..'):
    if not os.path.isfile(os.getcwd() + '/../' + path):
        sys.path.append(os.getcwd() + '/../' + path)
sys.path.append('../../libsvm-3.21/python')
sys.path.append('../../sketchfe/sketchfe')
import matplotlib.pyplot as plt
from extractor import *
from featureutil import *
import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import numpy as np
import operator
from draw import *
from LibSVM import *
import pickle
from Predictor import *
from fastCKMeans import *
from scipykmeans import *
from scipyCKMeans import *
from complexCKMeans import *

def processname(name):
    namesplit = name.split('_')
    sketchname = str(namesplit[0]+'_'+namesplit[1])
    partialnumber = namesplit[2] if len(namesplit) > 2 else 0

    return sketchname, partialnumber

def main():
    # count for the whole data to be loaded in to memory
    # including test and training

    K = [10]  # :O # number of cluster to test. can be a list
    # K = [numclass]
    N = range(1, 10)
    fullness = range(1, 11)
    import numpy as np
    C = np.linspace(0, 100, 51, endpoint=True)
    C = [int(c) for c in C]
    accuracy = dict()
    reject_rate = dict()
    total_answered = dict()

    sketchname2numpartial = dict()

    for k in K:
        for n in N:
            for c in C:
                accuracy[(k, n, c, True)] = 0
                accuracy[(k, n, c, False)] = 0
                reject_rate[(k, n, c, True)] = 0
                reject_rate[(k, n, c, False)] = 0

                total_answered[(k, n, c, False)] = 0
                total_answered[(k, n, c, True)] = 0

    classcounterlist = range(0,10)
    foldcounterlist = range(0,5)

    for classcounter in classcounterlist:
        print 'class: %i' % classcounter
        numclass, numfull, numpartial, numtest = 10, 50, 200, 10
        files = ['airplane', 'alarm-clock', 'angel', 'ant', 'apple', 'arm', 'armchair', 'ashtray', 'axe', 'backpack',
                 'banana',
                 'barn', 'baseball-bat', 'basket', 'bathtub', 'bear-(animal)', 'bed', 'bee', 'beer-mug', 'bell', 'bench',
                 'bicycle', 'binoculars', 'blimp', 'book', 'bookshelf', 'boomerang', 'bottle-opener', 'bowl', 'brain',
                 'bread',
                 'bridge', 'bulldozer', 'bus', 'bush', 'butterfly', 'cabinet', 'cactus', 'cake', 'calculator', 'camel',
                 'camera', 'candle', 'cannon', 'canoe', 'car-(sedan)', 'carrot', 'castle', 'cat', 'cell-phone', 'chair',
                 'chandelier',
                 'church', 'cigarette', 'cloud', 'comb', 'computer-monitor', 'computer-mouse', 'couch', 'cow', 'crab',
                 'crane-(machine)', 'crocodile', 'crown', 'cup', 'diamond', 'dog', 'dolphin', 'donut', 'door',
                 'door-handle',
                 'dragon', 'duck', 'ear', 'elephant', 'envelope', 'eye', 'eyeglasses', 'face', 'fan', 'feather',
                 'fire-hydrant',
                 'fish', 'flashlight', 'floor-lamp', 'flower-with-stem', 'flying-bird', 'flying-saucer', 'foot', 'fork',
                 'frog',
                 'frying-pan', 'giraffe', 'grapes', 'grenade', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat',
                 'head',
                 'head-phones', 'hedgehog', 'helicopter', 'helmet', 'horse', 'hot-air-balloon', 'hot-dog', 'hourglass',
                 'house',
                 'human-skeleton', 'ice-cream-cone', 'ipod', 'kangaroo', 'key', 'keyboard', 'knife', 'ladder', 'laptop',
                 'leaf',
                 'lightbulb', 'lighter', 'lion', 'lobster', 'loudspeaker', 'mailbox', 'megaphone', 'mermaid', 'microphone',
                 'microscope', 'monkey', 'moon', 'mosquito', 'motorbike', 'mouse-(animal)', 'mouth', 'mug', 'mushroom',
                 'nose',
                 'octopus', 'owl', 'palm-tree', 'panda', 'paper-clip', 'parachute', 'parking-meter', 'parrot', 'pear',
                 'pen',
                 'penguin', 'person-sitting', 'person-walking', 'piano', 'pickup-truck', 'pig', 'pigeon', 'pineapple',
                 'pipe-(for-smoking)', 'pizza', 'potted-plant', 'power-outlet', 'present', 'pretzel', 'pumpkin', 'purse',
                 'rabbit', 'race-car', 'radio', 'rainbow', 'revolver', 'rifle', 'rollerblades', 'rooster', 'sailboat',
                 'santa-claus', 'satellite', 'satellite-dish', 'saxophone', 'scissors', 'scorpion', 'screwdriver',
                 'sea-turtle',
                 'seagull', 'shark', 'sheep', 'ship', 'shoe', 'shovel', 'skateboard']


        # extract the whole data, including test and training
        from random import shuffle
        import random
        rnd = random.randint(1,10)
        files = files[classcounter*10:classcounter*10+10]
        extr = Extractor('../data/')
        whole_features, \
        whole_isFull, \
        whole_classId, \
        whole_names, \
        folderList = extr.loadfolders(
                                numclass,
                                numfull=numfull,
                                numpartial=numpartial,
                                folderList=files)

        # split data into train and test
        # for each class numtest full sketches -with all their partial sketches- are allocated for testing, and the rest
        # for training
        xfold_features,\
        xfold_isFull,\
        xfold_classId,\
        xfold_names =       xfoldpartition(
                                    max(foldcounterlist)+1,
                                    whole_features,
                                    whole_isFull,
                                    whole_classId,
                                    whole_names,
                                    folderList)


        for foldcounter in foldcounterlist:
            print 'fold: %i' %foldcounter
            train_features, train_isFull, train_classId, train_names = [],[],[],[]
            for c in foldcounterlist:
                if c != foldcounter:
                    train_features.extend(xfold_features[c])
                    train_isFull.extend(xfold_isFull[c])
                    train_classId.extend(xfold_classId[c])
                    train_names.extend(xfold_names[c])

            test_features = xfold_features[foldcounter]
            test_isFull = xfold_isFull[foldcounter]
            test_names = xfold_names[foldcounter]
            test_classId = xfold_classId[foldcounter]

            for k in K:
                '''
                Testing and training
                data is ready

                Training start
                '''

                ForceTrain = True
                folderName = '%s___%i_%i_%i_%i' % ('complexCKMeans_newprior-2', numclass, numfull, numpartial, k)
                trainingpath = '../data/training/' + folderName

                # if training data is already computed, import
                fio = FileIO()
                if os.path.exists(trainingpath) and not ForceTrain:
                    # can I assume consistency with classId and others?
                    _, _, _, _, kmeansoutput, _ = fio.loadTraining(
                        trainingpath, loadFeatures=False)
                    svm = SVM(kmeansoutput, train_classId, trainingpath + "/" + folderName, train_features)

                else:
                    # check if pycuda is installed
                    import imp
                    try:
                        imp.find_module('pycuda')
                        found = True
                    except ImportError:
                        found = False

                    if not found:
                        ckmeans = ComplexCKMeans(train_features, train_isFull, train_classId, k, maxiter=20)
                        kmeansoutput = ckmeans.getCKMeans()

                        trainer = Trainer(kmeansoutput, train_classId, train_features)
                        heteClstrFeatureId, heteClstrId = trainer.getHeterogenousClusterId()
                        fio.saveTraining(train_names, train_classId, train_isFull, train_features, kmeansoutput,
                                         trainingpath, folderName)
                        svm = trainer.trainSVM(heteClstrFeatureId, trainingpath)
                    else:
                        from complexCudaCKMeans import *
                        clusterer = complexCudaCKMeans(train_features, k, train_classId, train_isFull)  # FEATURES : N x 720
                        kmeansoutput = clusterer.cukmeans()

                        # find heterogenous clusters and train svm
                        trainer = Trainer(kmeansoutput, train_classId, train_features)
                        heteClstrFeatureId, heteClstrId = trainer.getHeterogenousClusterId()
                        fio.saveTraining(train_names, train_classId, train_isFull, train_features, kmeansoutput,
                                         trainingpath, folderName)
                        svm = trainer.trainSVM(heteClstrFeatureId, trainingpath)

                predictor = Predictor(kmeansoutput, train_classId, trainingpath, svm=svm)
                priorClusterProb = predictor.calculatePriorProb()

                print 'Starting Testing'
                # get predictions -class probabilities- for every test feature
                #classProbList = predictor.calculatePosteriorProb(test_features, priorClusterProb)
                for test_index in range(len(test_features)):
                    print 'Testing ' + str(test_index) + '(out of ' + str(len(test_features)) + ')'
                    Tfeature = test_features[test_index]
                    TtrueClass = test_classId[test_index]

                    classProb = predictor.calculatePosteriorProb(Tfeature, priorClusterProb)
                    #classProb = classProbList[test_index]
                    # sort the predictions -in the dictionary-
                    SclassProb = sorted(classProb.items(), key=operator.itemgetter(1))

                    for n in N:
                        for c in C:
                            # get the greatest n prediction
                            SPartialclassProb = SclassProb[-n:]
                            # sum first greatest n prediction and sum
                            summedprob = sum(tup[1] for tup in SPartialclassProb) * 100
                            # get the class id's of the greatest n prediction
                            summedclassId = [tup[0] for tup in SPartialclassProb]

                            #print SclassProb[0]

                            # if sum is less tran threshold C, reject
                            if summedprob < c:
                                reject_rate[(k, n, c, test_isFull[test_index])] += 1
                            else:
                                total_answered[(k, n, c, test_isFull[test_index])] += 1

                            # if not rejected and true class is inside the predictions, hit
                            if summedprob > c and TtrueClass in summedclassId:
                                # correct guess
                                accuracy[(k, n, c, test_isFull[test_index])] += 1

                print 'Partial Accuracy: %f'%(accuracy[(10, 1, 0, False)] * 1.0 / total_answered[(10, 1, 0, False)] * 100 if total_answered[(10, 1, 0, False)] != 0 else 0)
                print trainingpath + ' end'
            print 'Testing End'

        '''
        Calculate %
        '''

    for key in reject_rate:
        # normalize the reject rate
        reject_rate[key] = (reject_rate[key]*1.0/total_answered[key])*100 if total_answered[key] != 0 else 0

    for key in accuracy:
        # normalize the accuracy.
        # Count the number of non-rejected sketches to find true accuracy
        #total_un_answered = int(test_isFull.count(key[3])*(reject_rate[key]/100))
        #total_answered = test_isFull.count(key[3]) - total_un_answered
        accuracy[key] = (accuracy[key]*1.0/total_answered[key])*100 if total_answered[key] != 0 else 0

    '''
    Save results
    '''
    print 'Saving plots'
    pickle.dump(accuracy, open(trainingpath + '/' "accuracy.p", "wb"))
    pickle.dump(reject_rate, open(trainingpath + '/' "reject.p", "wb"))
    #pickle.dump(accuracy_fullness, open(trainingpath + '/' "accuracy_fullness.p", "wb"))

    draw_N_C_Acc(accuracy, N, C, k=K[0], isfull=True, path=trainingpath)
    draw_N_C_Reject_Contour(reject_rate, N, C, k=K[0], isfull=True, path=trainingpath)
    draw_N_C_Acc_Contour(accuracy, N, C, k=K[0], isfull=True, path=trainingpath)# Surface over n and c
    draw_N_C_Reject_Contour(reject_rate, N, C, k=K[0], isfull=True, path=trainingpath)
    draw_n_Acc(accuracy, c=0, k=K[0], isfull=True, reject_rate=reject_rate, path=trainingpath)# for fixed n and c
    #draw_K_Delay_Acc(accuracy, reject_rate, K=K, C=C, n=1, isfull=True, path=trainingpath)
    draw_Reject_Acc([accuracy], [reject_rate], N=[1, 2], k=K[0], isfull=True, labels=['Ck-means'], path=trainingpath)


    draw_N_C_Acc(accuracy, N, C, k=K[0], isfull=False, path=trainingpath)
    draw_N_C_Reject_Contour(reject_rate, N, C, k=K[0], isfull=False, path=trainingpath)
    draw_N_C_Acc_Contour(accuracy, N, C, k=K[0], isfull=False, path=trainingpath)# Surface over n and c
    draw_N_C_Reject_Contour(reject_rate, N, C, k=K[0], isfull=False, path=trainingpath)
    draw_n_Acc(accuracy, c=0, k=K[0], isfull=False, reject_rate=reject_rate, path=trainingpath)# for fixed n and c
    #draw_K_Delay_Acc(accuracy, reject_rate, K=K, C=C, n=1, isfull=False, path=trainingpath)
    draw_Reject_Acc([accuracy], [reject_rate], N=[1, 2], k=K[0], isfull=False, labels=['Ck-means'], path=trainingpath)

    #draw_Completeness_Accuracy(accuracy_fullness, fullness=fullness, k=K[0], n=1, C=np.linspace(0, 80, 5), isfull=False,
    #                           path=trainingpath)

    draw_N_C_Acc_Contour_Low(accuracy, N, C, k=K[0], isfull=False, path=trainingpath)

if __name__ == "__main__": main()