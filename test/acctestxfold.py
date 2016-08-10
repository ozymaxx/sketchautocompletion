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
import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import numpy as np
import operator
from draw import *
from SVM import *

def main():
    numclass, numfull, numpartial, xfold = 10, 10, 3, 2
    folderList = []

    extr = Extractor('../data/')
    whole_features, \
    whole_isFull, \
    whole_classId, \
    whole_names, \
    folderList = extr.loadfolders(
                            numclass,
                            numfull=numfull,
                            numpartial=numpartial)

    xfold_features, \
    xfold_isFull, \
    xfold_classId, \
    xfold_names =  xfoldpartition(
                        xfold,
                        whole_features,
                        whole_isFull,
                        whole_classId,
                        whole_names,
                        folderList)

    K = [numclass] # :O
    N = range(1, numclass)
    import numpy as np
    C = np.linspace(0, 100, 200, endpoint=False)
    accuracy = dict()
    delay_rate = dict()

    accuracySVM = dict()
    delay_rateSVM = dict()
    testcount = 0
    for k in K:
        for n in N:
            for c in C:
                accuracy[(k, n, c, True)] = 0
                accuracy[(k, n, c, False)] = 0
                delay_rate[(k, n, c, True)] = 0
                delay_rate[(k, n, c, False)] = 0

                accuracySVM[(k, n, c, True)] = 0
                accuracySVM[(k, n, c, False)] = 0
                delay_rateSVM[(k, n, c, True)] = 0
                delay_rateSVM[(k, n, c, False)] = 0


    for k in K:
        for it in range(xfold):
            test_features, test_isFull, test_classId, test_names = \
                xfold_features[it], xfold_isFull[it], xfold_classId[it], xfold_names[it]

            train_features, train_isFull, train_classId, train_names = [], [], [], []
            for trainIter in range(xfold):
                if trainIter != it:
                    train_features.extend(xfold_features[trainIter])
                    train_isFull.extend(xfold_isFull[trainIter])
                    train_classId.extend(xfold_classId[trainIter])
                    train_names.extend(xfold_names[trainIter])

            '''
            Testing and training
            data is ready
            '''

            '''
            Training start
            '''

            ForceTrain = True
            folderName = '%s__CFPK_%i_%i_%i_%i_%i' % ('xfold_cv_', numclass, numfull, numpartial, k, xfold)
            trainingName = '%s__CFPK_%i_%i_%i_%i_%i' % ('xfold_cv_' + str(it), numclass, numfull, numpartial, k, xfold)
            print trainingName

            trainingpath = '../data/training/' + folderName + '/' + trainingName

            # if training data is already computed, import
            fio = FileIO()
            if os.path.exists(trainingpath) and not ForceTrain:
                # can I assume consistency with classId and others ?
                _, _, _, _, kmeansoutput, _ = fio.loadTraining(
                    trainingpath + "/" + trainingName)
                svm = SVM(kmeansoutput, train_classId, trainingpath + "/" + trainingName, train_features)

            else:
                constarr = getConstraints(size=len(train_features), isFull=train_isFull, classId=train_classId)
                ckmeans = CKMeans(constarr, np.transpose(train_features), k)
                kmeansoutput = ckmeans.getCKMeans()

                # find heterogenous clusters and train svm
                trainer = Trainer(kmeansoutput, train_classId, train_features)
                heteClstrFeatureId, heteClstrId = trainer.getHeterogenous()
                fio.saveTraining(train_names, train_classId, train_isFull, train_features, kmeansoutput,
                                 trainingpath, trainingName)
                svm = trainer.trainSVM(heteClstrFeatureId, trainingpath)

            predictor = Predictor(kmeansoutput, train_classId, trainingpath, svm=None)

            priorClusterProb = predictor.calculatePriorProb()

            for test_index in range(len(test_features)):
                testcount += 1
                Tfeature = test_features[test_index]
                TtrueClass = test_classId[test_index]

                classProb = predictor.calculatePosteriorProb(Tfeature, priorClusterProb, numericKeys=True)
                SclassProb = sorted(classProb.items(), key=operator.itemgetter(1))

                for n in N:
                    for c in C:
                        SPartialclassProb = SclassProb[-n:]
                        summedprob = sum(tup[1] for tup in SPartialclassProb) * 100
                        summedclassId = [tup[0] for tup in SPartialclassProb]

                        if summedprob < c:
                            delay_rate[(k, n, c, test_isFull[test_index])] += 1
                        else:
                            if TtrueClass in summedclassId:
                                accuracy[(k, n, c, test_isFull[test_index])] += 1


            print trainingName + ' end'

    '''
    Calculate %
    '''

    for key in delay_rate:
        delay_rate[key] = (delay_rate[key]*1.0/whole_isFull.count(key[3]))*100

    for key in accuracy:
        total_un_answered = int(whole_isFull.count(key[3])*(delay_rate[key]/100))
        total_answered = whole_isFull.count(key[3]) - total_un_answered
        accuracy[key] = (accuracy[key]*1.0/total_answered)*100 if total_answered != 0 else 100

    '''
    Save results and load back
    '''

    draw_N_C_Acc(accuracy, N, C, k=numclass, isfull=True)
    draw_N_C_Reject_Contour(delay_rate, N, C, k=numclass, isfull=True)
    draw_N_C_Acc_Contour(accuracy, N, C, k=numclass, isfull=True) # Surface over n and c
    draw_N_C_Reject_Contour(delay_rate, N, C, k=numclass, isfull=True)
    draw_n_Acc(accuracy, c=30, k=numclass, isfull=False, delay_rate=delay_rate) # for fixed n and c
    draw_K_Delay_Acc(accuracy, delay_rate, K=K, C=C, n=1, isfull=True)
    draw_Reject_Acc([accuracy], [delay_rate], N=[1, 2], k=k, isfull=True, labels=['Ck-means'])
    draw_K-C-Text_Acc(accuracy, delay_rate, 'Constrained Voting')
if __name__ == "__main__": main()