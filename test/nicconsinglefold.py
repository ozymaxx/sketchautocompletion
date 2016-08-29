import sys
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append('../data/')
sys.path.append("../../libsvm-3.21/python/")
import matplotlib.pyplot as plt
from extractor import *
from featureutil import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import numpy as np
import operator
from draw import *
from SVM import *
import pickle
from Predictor import *
from scipyCKMeans import *
from scipykmeans import *
from complexCKMeans import *
from complexcudackmeans import *

def main():
    numclass = 14

    extr_train = Extractor('C:/Users/1003/Desktop/nicicon/csv/train')
    train_features, \
    train_isFull, \
    train_classId, \
    train_names = extr_train.loadniciconfolders()

    extr_test = Extractor('C:/Users/1003/Desktop/nicicon/csv/test')
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

        ForceTrain = True
        folderName = '%s___%i_%i' % ('nicicionWithComplexCKMeans_fulldata_newprior', max(train_classId)+1, k)
        trainingpath = '../data/training/' + folderName

        # if training data is already computed, import
        fio = FileIO()
        if os.path.exists(trainingpath) and not ForceTrain:
            # can I assume consistency with classId and others ?
            _, _, _, _, kmeansoutput, _ = fio.loadTraining(
                trainingpath + '/' + folderName)
            svm = SVM(kmeansoutput, train_classId, trainingpath, train_features)
            svm.loadModels()

        else:
            # check if pycuda is installed
            import imp
            try:
                imp.find_module('pycuda')
                found = True
            except ImportError:
                found = False

            if not found:
                ckmeans = complexCKMeans(train_features, train_isFull, train_classId, k, maxiter=25)
                kmeansoutput = ckmeans.getCKMeans()

                # find heterogenous clusters and train svm
                trainer = Trainer(kmeansoutput, train_classId, train_features)
                heteClstrFeatureId, heteClstrId = trainer.getHeterogenousClusterId()
                fio.saveTraining(train_names, train_classId, train_isFull, train_features, kmeansoutput,
                                 trainingpath, folderName)
                svm = trainer.trainSVM(heteClstrFeatureId, trainingpath)
            else:
                from cudackmeans import *
                clusterer = complexCudaCKMeans(train_features, k, train_classId, train_isFull)  # FEATURES : N x 720
                # print 'pycuda', timeit.timeit(lambda: clusterer.cukmeans(data, clusters), number=rounds)
                clusters, centers = clusterer.cukmeans()
                kmeansoutput = [clusters, centers]

                # find heterogenous clusters and train svm
                trainer = Trainer(kmeansoutput, train_classId, train_features)
                heteClstrFeatureId, heteClstrId = trainer.getHeterogenousClusterId()
                fio.saveTraining(train_names, train_classId, train_isFull, train_features, kmeansoutput,
                                 trainingpath, folderName)
                svm = trainer.trainSVM(heteClstrFeatureId, trainingpath)

        predictor = Predictor(kmeansoutput, train_classId, trainingpath, svm=svm)
        priorClusterProb = predictor.calculatePriorProb()

        classProbList = predictor.calculatePosteriorProb(test_features, priorClusterProb)
        print 'Starting Testing'
        classProbList = predictor.calculatePosteriorProb(test_features, priorClusterProb)
        for test_index in range(len(test_features)):
            print 'Testing ' + str(test_index) + '(out of ' + str(len(test_features)) + ')'
            Tfeature = test_features[test_index]
            TtrueClass = test_classId[test_index]

            #classProb = predictor.calculatePosteriorProb(Tfeature, priorClusterProb)
            classProb = classProbList[test_index]
            SclassProb = sorted(classProb.items(), key=operator.itemgetter(1))

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
    print 'Testing End'

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
    draw_n_Acc(accuracy, c=0, k=K[0], isfull=True, delay_rate=reject_rate, path=trainingpath)# for fixed n and c
    #draw_K_Delay_Acc(accuracy, reject_rate, K=K, C=C, n=1, isfull=True, path=trainingpath)
    draw_Reject_Acc([accuracy], [reject_rate], N=[1, 2], k=K[0], isfull=True, labels=['Ck-means'], path=trainingpath)

    draw_N_C_Acc(accuracy, N, C, k=K[0], isfull=False, path=trainingpath)
    draw_N_C_Reject_Contour(reject_rate, N, C, k=K[0], isfull=False, path=trainingpath)
    draw_N_C_Acc_Contour(accuracy, N, C, k=K[0], isfull=False, path=trainingpath)# Surface over n and c
    draw_N_C_Reject_Contour(reject_rate, N, C, k=K[0], isfull=False, path=trainingpath)
    draw_n_Acc(accuracy, c=0, k=K[0], isfull=False, delay_rate=reject_rate, path=trainingpath)# for fixed n and c
    #draw_K_Delay_Acc(accuracy, reject_rate, K=K, C=C, n=1, isfull=False, path=trainingpath)
    draw_Reject_Acc([accuracy], [reject_rate], N=[1, 2], k=K[0], isfull=False, labels=['Ck-means'], path=trainingpath)

    #draw_K-C-Text_Acc(accuracy, reject_rate, 'Constrained Voting')
if __name__ == "__main__": main()