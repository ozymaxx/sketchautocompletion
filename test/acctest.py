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
import os
import operator

def trainIt(trainingName, trainingpath, numclass, numfull, numpartial, k, files):

    constarr = getConstraints(size=len(features), isFull=isFull, classId=classId)
    ckmeans = CKMeans(constarr, np.transpose(features), k)
    kmeansoutput = ckmeans.getCKMeans()

    # find heterogenous clusters and train svm
    trainer = Trainer(kmeansoutput, classId, features)
    heteClstrFeatureId, heteClstrId = trainer.getHeterogenous()
    fio.saveTraining(names, classId, isFull, features, kmeansoutput,
                     trainingpath, trainingName)
    trainer.trainSVM(heteClstrFeatureId, trainingpath)
    return kmeansoutput, classId

def main():
    '''
    ForceTrain = False
    numclass, numfull, numpartial = 10, 10, 3
    k = numclass

    trainingName = '%s__CFPK_%i_%i_%i_%i' % ('training', numclass, numfull, numpartial, k)
    trainingpath = '../data/training/' + trainingName

    fio = FileIO()
    extr = Extractor('../data/')

    # if training data is already computed, import
    if os.path.exists(trainingpath) and not ForceTrain:

        train_names, \
        train_classId, \
        train_isFull, \
        train_features, \
        kmeansoutput, \
        folderList = fio.loadTraining(trainingpath + "/" + trainingName)

        whole_features, \
        whole_isFull, \
        whole_classId, \
        whole_names, \
        folderList = extr.loadfolders(
                              numclass,
                              numfull    = 80,
                              numpartial = 80,
                              folderList = folderList)
        train_features,\
        train_isFull,\
        train_classId,\
        train_names,\
        test_features,\
        test_isFull,\
        test_names,\
        test_classid = partitionfeatures_notin(
                                    whole_features,
                                    whole_isFull,
                                    whole_classId,
                                    whole_names,
                                    numtestfull,
                                    numtestpartial,
                                    train_names,
                                    selectTestRandom=True)
    else: # else train


        whole_features,\
        whole_isFull,\
        whole_classId,\
        whole_names,\
        whole_folderList = extr.loadfolders(
                                    numclass   = numclass,
                                    numfull    = numfull,
                                    numpartial = numpartial,
                                    folderList = [])

        train_features,\
        train_isFull,\
        train_classId,\
        train_names,\
        test_features,\
        test_isFull,\
        test_names,\
        test_classid = partitionfeatures(
                                    whole_features,
                                    whole_isFull,
                                    whole_classId,
                                    whole_names,
                                    numtestfull,
                                    numtestpartial,
                                    selectTestRandom=True)

        constarr = getConstraints(size=len(train_features), isFull=train_isFull, classId=train_classId)
        ckmeans = CKMeans(constarr, np.transpose(train_features), k)
        kmeansoutput = ckmeans.getCKMeans()

        # find heterogenous clusters and train svm
        trainer = Trainer(kmeansoutput, train_classId, train_features)
        heteClstrFeatureId, heteClstrId = trainer.getHeterogenous()
        fio.saveTraining(train_names, train_classId, train_isFull, train_features, kmeansoutput,
                         trainingpath, trainingName)
        trainer.trainSVM(heteClstrFeatureId, trainingpath)

    predictor = Predictor(kmeansoutput, train_classId, trainingpath)
    priorClusterProb = predictor.calculatePriorProb()


    '''
    numclass, numfull, numpartial, xfold = 5, 5, 5, 5

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


    K = [xfold] # :O
    N = range(2, 3)
    C = range(30, 31)
    accuracy = dict()
    delay_rate = dict()

    for k, n, c in itertools.izip(K, N, C):
        accuracy[(k, n, c, True)] = 0
        accuracy[(k, n, c, False)] = 0
        delay_rate[(k, n, c, True)] = 0
        delay_rate[(k, n, c, False)] = 0

    for k in K:
        for it in range(xfold):
            test_features, test_isFull, test_classId, test_names = \
                xfold_features[it], xfold_isFull[it], xfold_classId[it], xfold_names[it]

            train_features, train_isFull, train_classId, train_names = [],[],[],[]
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

            ForceTrain = False
            trainingName = '%s__CFPK_%i_%i_%i_%i_%i' % ('xfold_cv_' + str(it), numclass, numfull, numpartial, k, xfold)
            print trainingName
            trainingpath = '../data/training/' + trainingName

            # if training data is already computed, import
            fio = FileIO()
            if os.path.exists(trainingpath) and not ForceTrain:
                # can I assume consistency with classId and others ?
                _, _, _, _, kmeansoutput, _ = fio.loadTraining(
                    trainingpath + "/" + trainingName)
            else:
                constarr = getConstraints(size=len(train_features), isFull=train_isFull, classId=train_classId)
                ckmeans = CKMeans(constarr, np.transpose(train_features), k)
                kmeansoutput = ckmeans.getCKMeans()

                # find heterogenous clusters and train svm
                trainer = Trainer(kmeansoutput, train_classId, train_features)
                heteClstrFeatureId, heteClstrId = trainer.getHeterogenous()
                fio.saveTraining(train_names, train_classId, train_isFull, train_features, kmeansoutput,
                                 trainingpath, trainingName)
                trainer.trainSVM(heteClstrFeatureId, trainingpath)

            predictor = Predictor(kmeansoutput, train_classId, trainingpath)
            priorClusterProb = predictor.calculatePriorProb()

            for test_index in range(len(test_features)):
                Tfeature = test_features[test_index]
                TtrueClass = test_classId[test_index]
                classProb = predictor.calculatePosteriorProb(Tfeature, priorClusterProb)

                SclassProb = sorted(classProb.items(), key=operator.itemgetter(1))
                SclassProb = SclassProb[-n:]

                summedprob = sum(tup[1] for tup in SclassProb)*100
                summedclassId = [tup[0] for tup in SclassProb]

                for n, c in itertools.izip(N, C):
                    if summedprob < c:
                        delay_rate[(k, n, c, test_isFull[test_index])] += 1
                    else:
                        if TtrueClass in summedclassId:
                            accuracy[(k, n, c, test_isFull[test_index])] += 1

            print trainingName + ' end'

        for key in accuracy:
            accuracy[key] = accuracy[key]*1.0/len(whole_features)*100

        for key in delay_rate:
            delay_rate[key] = delay_rate[key]*1.0/len(whole_features)*100





    '''
                
    ncount = [0]*numclass
    for index in range(len(test_features)):
        feature = test_features[index]
        name = test_names[index]

        maxClass = max(classProb, key=classProb.get)
        argsort = np.argsort(classProb.values())

        # calculate the accuracy
        ncounter = None
        for ncounter in range(numclass-1, -1, -1):
            if argsort[ncounter] == test_classid[index]:
                break

        while ncounter >= 0:
            ncount[len(ncount) - ncounter - 1] += 1
            ncounter += -1

    # change it to percentage
    print '# Class: %i \t# Full: %i \t# Partial: %i \t# Test case: %i' % (numclass, numfull, numpartial, numclass*numtestfull)
    ncount = [(x * 1.0 / (numtestfull * numclass)) * 100 for x in ncount]
    for accindex in range(min(10, len(ncount))):
        print 'N=' + str(accindex+1) + ' C=0 accuracy: ' + str(ncount[accindex])

    plt.plot(range(1,len(ncount)+1), ncount, 'bo')

    for x, y in itertools.izip(range(1, len(ncount)+1), ncount):
        plt.annotate('%i,%i%%'%(x, y), (x+0.1, y+0.1))

    plt.plot([1, len(ncount)+1], [60, 60], 'r--')
    plt.plot([1, len(ncount) + 1], [80, 80], 'g--')
    plt.ylabel('Accuracy %')
    plt.xlabel('N=')
    plt.title('Sketch: %i # Class: %i   # Full: %i   # Partial: %i # Test case: %i' % (len(train_features), numclass, numtestfull, numpartial, numclass*numtestfull))
    plt.ylim([0, 105])
    plt.xlim([1, len(ncount)+1])
    plt.grid(True)
    plt.show()

    '''
if __name__ == "__main__": main()