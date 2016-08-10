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
import pickle

def main():
    numclass, numfull, numpartial = 4, 80, 2
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

    train_features, train_isFull, train_classId, train_names, test_features, test_isFull, test_names, test_classId = \
        partitionfeatures(whole_features,
                          whole_isFull,
                          whole_classId,
                          whole_names,
                          numtrainfull = 70,
                          selectTestRandom=True)

    #K = range(numclass, numclass*2) # :O
    K = [numclass]
    N = range(1, numclass)
    import numpy as np
    C = np.linspace(0, 100, 200, endpoint=False)
    accuracy = dict()
    reject_rate = dict()

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

                accuracySVM[(k, n, c, True)] = 0
                accuracySVM[(k, n, c, False)] = 0
                delay_rateSVM[(k, n, c, True)] = 0
                delay_rateSVM[(k, n, c, False)] = 0

    for k in K:
        '''
        Testing and training
        data is ready
        '''

        '''
        Training start
        '''

        ForceTrain = True
        folderName = '%s___%i_%i_%i_%i' % ('singlefoldacctest', numclass, numfull, numpartial, k)
        trainingpath = '../data/training/' + folderName

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
                             trainingpath, folderName)
            svm = trainer.trainSVM(heteClstrFeatureId, trainingpath)

        predictor = Predictor(kmeansoutput, train_classId, trainingpath, svm=svm)

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
                        reject_rate[(k, n, c, test_isFull[test_index])] += 1

                    if summedprob > c and TtrueClass in summedclassId:
                            accuracy[(k, n, c, test_isFull[test_index])] += 1

        print trainingpath + ' end'

    '''
    Calculate %
    '''

    for key in reject_rate:
        reject_rate[key] = (reject_rate[key]*1.0/test_isFull.count(key[3]))*100

    for key in accuracy:
        total_un_answered = int(test_isFull.count(key[3])*(reject_rate[key]/100))
        total_answered = test_isFull.count(key[3]) - total_un_answered
        accuracy[key] = (accuracy[key]*1.0/total_answered)*100 if total_answered != 0 else -100

    '''
    Save results
    '''

    pickle.dump(accuracy, open(trainingpath + '/' "accuracy.p", "wb"))
    pickle.dump(reject_rate, open(trainingpath + '/' "reject.p", "wb"))

    draw_N_C_Acc(accuracy, N, C, k=numclass, isfull=True)
    draw_N_C_Reject_Contour(reject_rate, N, C, k=numclass, isfull=True)
    draw_N_C_Acc_Contour(accuracy, N, C, k=numclass, isfull=True) # Surface over n and c
    draw_N_C_Reject_Contour(reject_rate, N, C, k=numclass, isfull=True)
    draw_n_Acc(accuracy, c=30, k=numclass, isfull=False, delay_rate=reject_rate) # for fixed n and c
    draw_K_Delay_Acc(accuracy, reject_rate, K=K, C=C, n=1, isfull=True)
    draw_Reject_Acc([accuracy], [reject_rate], N=[1, 2], k=k, isfull=True, labels=['Ck-means'])
    #draw_K-C-Text_Acc(accuracy, reject_rate, 'Constrained Voting')
if __name__ == "__main__": main()