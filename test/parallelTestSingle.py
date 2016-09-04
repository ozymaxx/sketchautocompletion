"""
Ahmet BAGLAN
"""
import time
import sys
sys.path.append("../ParallelMethod/")
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append('../data/')
sys.path.append('../SVM/')
sys.path.append("../../libsvm-3.21/python/")

import matplotlib.pyplot as plt
from extractor import *
import os
import operator
from draw import *
from LibSVM import *
import pickle
from ParallelPredictorMaster import *
from ParallelTrainer import *
from FileIO import *
import parallelPartitioner
from centers import *


def main():
    doPartition = True
    numclass, numfull, numpartial = 40, 80, 80
    numtest = 10
    debugMode = False
    doKMeansGrouping = True
    groupByN = 6
    my_name = 'ParalelEatzCentersGroupBy6KMeansGroups40_80_80'
    K = [20] # :

    if doPartition:
        #Divide Data Save testing data to testingData Folder, trainingData to trainingData folder
        parallelPartitioner.partition(numclass, numfull, numpartial,numtest)

        print "CALCULATING CLASS CENTERS"
        saveCenters(doOnlyFulls= True, getDataFrom = "../data/trainingData/csv/" , numClass = numclass)

    files = ['airplane', 'alarm-clock', 'angel', 'ant', 'apple', 'arm', 'armchair', 'ashtray', 'axe',
             'backpack', 'banana', 'barn', 'baseball-bat', 'basket', 'bathtub', 'bear-(animal)', 'bed',
             'bee', 'beer-mug', 'bell', 'bench', 'bicycle', 'binoculars', 'blimp', 'book', 'bookshelf',
             'boomerang', 'bottle-opener', 'bowl', 'brain', 'bread', 'bridge', 'bulldozer', 'bus', 'bush',
             'butterfly', 'cabinet', 'cactus', 'cake', 'calculator', 'camel', 'camera', 'candle', 'cannon',
             'canoe', 'car-(sedan)', 'carrot', 'castle', 'cat', 'cell-phone', 'chair', 'chandelier',
             'church', 'cigarette', 'cloud', 'comb', 'computer-monitor', 'computer-mouse', 'couch', 'cow',
             'crab', 'crane-(machine)', 'crocodile', 'crown', 'cup', 'diamond', 'dog', 'dolphin', 'donut',
             'door', 'door-handle', 'dragon', 'duck', 'ear', 'elephant', 'envelope', 'eye', 'eyeglasses',
             'face', 'fan', 'feather', 'fire-hydrant', 'fish', 'flashlight', 'floor-lamp', 'flower-with-stem',
             'flying-bird', 'flying-saucer', 'foot', 'fork', 'frog', 'frying-pan', 'giraffe', 'grapes',
             'grenade', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'head', 'head-phones',
             'hedgehog', 'helicopter', 'helmet', 'horse', 'hot-air-balloon', 'hot-dog', 'hourglass',
             'house', 'human-skeleton', 'ice-cream-cone', 'ipod', 'kangaroo', 'key', 'keyboard', 'knife',
             'ladder', 'laptop', 'leaf', 'lightbulb', 'lighter', 'lion', 'lobster', 'loudspeaker',
             'mailbox', 'megaphone', 'mermaid', 'microphone', 'microscope', 'monkey', 'moon', 'mosquito',
             'motorbike', 'mouse-(animal)', 'mouth', 'mug', 'mushroom', 'nose', 'octopus', 'owl',
             'palm-tree', 'panda', 'paper-clip', 'parachute', 'parking-meter', 'parrot', 'pear', 'pen',
             'penguin', 'person-sitting', 'person-walking', 'piano', 'pickup-truck', 'pig', 'pigeon',
             'pineapple', 'pipe-(for-smoking)', 'pizza', 'potted-plant', 'power-outlet', 'present',
             'pretzel', 'pumpkin', 'purse', 'rabbit', 'race-car', 'radio', 'rainbow', 'revolver',
             'rifle', 'rollerblades', 'rooster', 'sailboat', 'santa-claus', 'satellite', 'satellite-dish',
             'saxophone', 'scissors', 'scorpion', 'screwdriver', 'sea-turtle', 'seagull', 'shark',
             'sheep', 'ship', 'shoe', 'shovel', 'skateboard', 'skull', 'skyscraper', 'snail', 'snake',
             'snowboard', 'snowman', 'socks', 'space-shuttle', 'speed-boat', 'spider', 'sponge-bob',
             'spoon', 'squirrel', 'standing-bird', 'stapler', 'strawberry', 'streetlight', 'submarine',
             'suitcase', 'sun', 'suv', 'swan', 'sword', 'syringe', 't-shirt', 'table', 'tablelamp',
             'teacup', 'teapot', 'teddy-bear', 'telephone', 'tennis-racket', 'tent', 'tiger', 'tire',
             'toilet', 'tomato', 'tooth', 'toothbrush', 'tractor', 'traffic-light', 'train', 'tree',
             'trombone', 'trousers', 'truck', 'trumpet', 'tv', 'umbrella', 'van', 'vase', 'violin',
             'walkie-talkie', 'wheel', 'wheelbarrow', 'windmill', 'wine-bottle', 'wineglass', 'wrist-watch',
             'zebra']

    myfio  = FileIO()
    extr = Extractor('../data/testingData/')#We will get our testing data from testingData folder


    test_features, \
    test_isFull, \
    test_classId, \
    test_names, \
    folderList = extr.loadFoldersParallel(#I have written this func this just gives without caring if the sketch is nearly full or what
                            numclass,
                            numfull=numfull,
                            numpartial=numpartial,
                            folderList=files)

    if(debugMode):
        print "Test Names As Follows"
        print test_names

    #After this Semih knows
    # K = [numclass] # :O
    #K = [numclass]
    N = range(1, numclass)
    import numpy as np
    C = np.linspace(0, 100, 51, endpoint=True)
    C = [int(c) for c in C]
    accuracy = dict()
    reject_rate = dict()
    my_files = folderList
    accuracySVM = dict()
    delay_rateSVM = dict()
    testcount = 0
    for k in K:
        for n in N:
            for c in C:
                accuracy[(k, n, c, True)] = 0
                accuracy[(k, n, c, False)] = 0
                reject_rate[(k, n, c, True)] = 0
                reject_rate[(k, n, c, False)] = 0

    for k in K:
        '''
        Testing and training
        data is ready
        '''

        '''
        Training start
        '''

        ForceTrain = False
        folderName = my_name
        trainingpath = '../data/newMethodTraining/' + my_name
        timeForTraining = "NOT CALC"
        # if training data is already computed, import
        fio = FileIO()
        if os.path.exists(trainingpath) and not ForceTrain:
            nameOfTheTraining = my_name
            predictor = ParallelPredictorMaster(nameOfTheTraining)

        else:
            start_time = time.time()
            myParallelTrainer = ParallelTrainer (groupByN,my_files, doKMeans = doKMeansGrouping)
            myParallelTrainer.trainSVM(numclass, numfull, numpartial, k, my_name)
            nameOfTheTraining = my_name
            predictor = ParallelPredictorMaster(nameOfTheTraining)
            stop_time = time.time()
            timeForTraining = stop_time-start_time
            file_ = open(trainingpath+'/TraingTime.txt', 'w')
            file_.write("Time elapsed for training = " + str(timeForTraining))
            file_.close()


        print 'Starting Testing'
        """
        lastTrueClass = ''
        test_features = test_features[1000:6000]
        test_isFull = test_isFull[1000:6000]
        test_classId =test_classId[1000:6000]
        test_names = test_names[1000:6000]
        """
        for test_index in range(len(test_features)):
            # print 'Testing ' + str(test_index) + '(out of ' + str(len(test_features)) + ')'
            Tfeature = test_features[test_index]
            TtrueClass = folderList[test_classId[test_index]]
            if debugMode:
                print "--------------------------------------------------"

            if(test_index%100 == 0):
                print  "Testing ", test_index, "(Out of ", len(test_features),")"
                s= time.time()
                classProb = predictor.calculatePosteriorProb(Tfeature)
                SclassProb = sorted(classProb.items(), key=operator.itemgetter(1))
                e = time.time()
                print "Prediction time = ", e-s
            else:
                classProb = predictor.calculatePosteriorProb(Tfeature)
                SclassProb = sorted(classProb.items(), key=operator.itemgetter(1))


            if debugMode:
                print "nowSclass", SclassProb,TtrueClass
                print "---------------------------------------------------"
            if debugMode:
                print SclassProb,"for the ans", TtrueClass

            for n in N:
                for c in C:
                    SPartialclassProb = SclassProb[-n:]
                    summedprob = sum(tup[1] for tup in SPartialclassProb) * 100
                    summedclassId = [tup[0] for tup in SPartialclassProb]

                    if summedprob < c:
                        reject_rate[(k, n, c, test_isFull[test_index])] += 1

                    if summedprob > c and TtrueClass in summedclassId:
                        accuracy[(k, n, c, test_isFull[test_index])] += 1

        print trainingpath + ' end'
    print 'Testint End'

    '''
    Calculate %
    '''

    for key in reject_rate:
        reject_rate[key] = (reject_rate[key]*1.0/test_isFull.count(key[3]))*100

    for key in accuracy:
        total_un_answered = int(test_isFull.count(key[3])*(reject_rate[key]/100))
        total_answered = test_isFull.count(key[3]) - total_un_answered
        accuracy[key] = (accuracy[key]*1.0/total_answered)*100 if total_answered != 0 else 0

    '''
    Save results
    '''
    print 'Saving plots'
    print trainingpath
    pickle.dump(accuracy, open(trainingpath + '/' "accuracy.p", "wb"))
    pickle.dump(reject_rate, open(trainingpath + '/' "reject.p", "wb"))

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
    print "Training Time = ", timeForTraining
    #draw_K-C-Text_Acc(accuracy, reject_rate, 'Constrained Voting')
if __name__ == "__main__": main()
