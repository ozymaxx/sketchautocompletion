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
def main():
    ForceTrain = False
    numclass, numfull, numpartial, numtest = 10, 10, 3, 5
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
                              numfull=numfull+numtest,
                              numpartial= numpartial)
        trainfeatures,\
        trainisFull,\
        trainclassId,\
        trainnames,\
        testfeatures,\
        testisFull,\
        testnames,\
        testclassid = partitionfeatures_notin(
                                    whole_features,
                                    whole_isFull,
                                    whole_classId,
                                    whole_names,
                                    numtest,
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
                                    numtest,
                                    selectTestRandom=True)

        constarr = getConstraints(size=len(train_features), isFull=train_isFull, classId=train_classId)
        ckmeans = CKMeans(constarr, np.transpose(train_features), k)
        kmeansoutput = ckmeans.getCKMeans()

        # find heterogenous clusters and train svm
        trainer = Trainer(kmeansoutput, train_classId, train_features)
        heteClstrFeatureId, heteClstrId = trainer.getHeterogenous()
        fio.saveTraining(names, classId, isFull, features, kmeansoutput,
                         trainingpath, trainingName)
        trainer.trainSVM(heteClstrFeatureId, trainingpath)

    predictor = Predictor(kmeansoutput, train_classId, trainingpath)

    '''

    Test and training data is ready
    Predictor is ready

    '''

    N = range(1, 10)
    C = range(30, 5, 50)

    ncount = [0]*numclass
    for index in range(len(testfeatures)):
        feature = testfeatures[index]
        name = testnames[index]

        priorClusterProb = predictor.calculatePriorProb()
        classProb = predictor.calculatePosteriorProb(feature, priorClusterProb)
        maxClass = max(classProb, key=classProb.get)
        argsort = np.argsort(classProb.values())

        # calculate the accuracy
        ncounter = None
        for ncounter in range(numclass-1, -1, -1):
            if argsort[ncounter] == testclassid[index]:
                break

        while ncounter >= 0:
            ncount[len(ncount) - ncounter - 1] += 1
            ncounter += -1

    # change it to percentage
    print '# Class: %i \t# Full: %i \t# Partial: %i \t# Test case: %i' % (numclass, numfull, numpartial, numclass*numtest)
    ncount = [(x * 1.0 / (numtest * numclass)) * 100 for x in ncount]
    for accindex in range(min(10, len(ncount))):
        print 'N=' + str(accindex+1) + ' C=0 accuracy: ' + str(ncount[accindex])

    plt.plot(range(1,len(ncount)+1), ncount, 'bo')

    for x, y in itertools.izip(range(1, len(ncount)+1), ncount):
        plt.annotate('%i,%i%%'%(x, y), (x+0.1, y+0.1))

    plt.plot([1, len(ncount)+1], [60, 60], 'r--')
    plt.plot([1, len(ncount) + 1], [80, 80], 'g--')
    plt.ylabel('Accuracy %')
    plt.xlabel('N=')
    plt.title('Sketch: %i # Class: %i   # Full: %i   # Partial: %i # Test case: %i' % (len(train_features), numclass, numfull, numpartial, numclass*numtest))
    plt.ylim([0, 105])
    plt.xlim([1, len(ncount)+1])
    plt.grid(True)
    plt.show()
if __name__ == "__main__": main()