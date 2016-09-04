import sys
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append('../data/')
sys.path.append('../ParallelMethod/')
sys.path.append('../SVM/')
sys.path.append("../../libsvm-3.21/python/")
from extractor import *
import os
import numpy as np
import operator
from draw import *
from LibSVM import *
import pickle
from Predictor import *
from scipyCKMeans import *
from scipykmeans import *
from complexCKMeans import *
from ParallelPredictorMaster import *
from ParallelTrainer import *
import grouper
import time
def main():

    files = ['accident','bomb','car','casualty','electricity','fire','firebrigade','flood','gas','injury','paramedics','person','police','roadblock']
    numclass = 14
    numfull = 80
    numpartial = 80
    doKMeansGrouping = False
    numOfGroups = 2
    my_name = 'ParallelNICGroupBy5RandomGroups'
    debugMode = False

    trainingFolder = '../data/nicicon/csv/train'
    extr_train = Extractor(trainingFolder)
    train_features, \
    train_isFull, \
    train_classId, \
    train_names = extr_train.loadniciconfolders()

    extr_test = Extractor('../data/nicicon/csv/test')
    test_features, \
    test_isFull, \
    test_classId, \
    test_names = extr_test.loadniciconfolders()

    K = [20] # :O
    #K = [numclass]
    N = range(1, numclass)
    import numpy as np
    C = np.linspace(0, 100, 51, endpoint=True)
    C = [int(c) for c in C]
    accuracy = dict()
    reject_rate = dict()
    my_files = files#list(set(test_classId))


    for k in K:
        for n in N:
            for c in C:
                accuracy[(k, n, c, True)] = 0
                accuracy[(k, n, c, False)] = 0
                reject_rate[(k, n, c, True)] = 0
                reject_rate[(k, n, c, False)] = 0

    for k in K:
        if debugMode:
            print '------------------------------------------------------------------'
            print '------------------------------------------------------------------'
            print 'K = ', k
            print '------------------------------------------------------------------'
            print '------------------------------------------------------------------'
            print '------------------------------------------------------------------'

        '''
        Testing and training
        data is ready
        '''

        '''
        Training start
        '''

        ForceTrain = False
        folderName = '%s___%i_%i' % ('nicicionWithComplexCKMeans_fulldata_newprior', max(test_classId)+1, k)
        trainingpath = '../data/newMethodTraining/' + my_name+'/'+ folderName

        # if training data is already computed, import
        fio = FileIO()
        if os.path.exists(trainingpath) and not ForceTrain:
            print 'lol'

        else:

            if not os.path.exists(trainingpath):
                 os.makedirs(trainingpath)
            start_time = time.time()
            if doKMeansGrouping:
                groups = grouper.doKMeansGrouping(numOfGroups,files[:numclass],centersFolder = '../data/newMethodTraining/allCentersNic.csv')
            else:
                groups = grouper.doRandomGrouping(numOfGroups,files[:numclass])

            klist = grouper.kListGenerator(groups)
            myParallelTrainer = ParallelTrainer (my_name, groups, klist, getTrainingDataFrom = '../data/nicicon/csv/train/')
            myParallelTrainer.trainSVM(numclass, numfull, numpartial)
            stop_time = time.time()
            timeForTraining = stop_time-start_time
            file_ = open(trainingpath+'/TraingTime.txt', 'w')
            file_.write("Time elapsed for training = " + str(timeForTraining))
            file_.close()

            # myParallelTrainer = ParallelTrainer (groupByN,my_files, doKMeans = doKMeansGrouping, getTrainingDataFrom = , centersFolder =  '../data/newMethodTraining/allCentersNic.csv')
            # myParallelTrainer.trainSVM(numclass, numfull, numpartial, k, my_name)
        if debugMode:
            print 'now first training is done ------------------------------------------++++++++++++++++++'
        nameOfTheTraining = my_name
        predictor = ParallelPredictorMaster(nameOfTheTraining)

        print 'Starting Testing'
        start_timeTest = time.time()
        for test_index in range(len(test_features)):
            print 'Testing ' + str(test_index) + '(out of ' + str(len(test_features)) + ')'
            Tfeature = test_features[test_index]
            TtrueClass = files[test_classId[test_index]]

            for i in files:
                if(i in test_names[test_index]):
                    TtrueClass = i

            classProb = predictor.calculatePosteriorProb(Tfeature)
            SclassProb = sorted(classProb.items(), key=operator.itemgetter(1))
            # print "nowSclass", SclassProb,TtrueClass,test_names[test_index]
            # print "--------------------------------------------------"


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
    print 'Testing End Time Elapsed : ', (time.time()-start_timeTest)
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

    #draw_K-C-Text_Acc(accuracy, reject_rate, 'Constrained Voting')
if __name__ == "__main__": main()
