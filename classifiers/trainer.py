"""
Classifier trainer
Ahmet BAGLAN - Arda ICMEZ - Semih GUNEL
14.07.2016
"""

import sys
sys.path.append("../clusterer/")
sys.path.append("../../libsvm-3.21/python/")

from svmutil import *
from numpy.random import rand
import numpy as np
import matplotlib.pyplot as plt
from ckmeans import *
from getConstraints import *
import visualise

def trainSVM(featArr, clusArr, labArr) :

    #label = labArr.tolist()
    label = labArr
    order = 0
    allModels = list()
    for i in clusArr:
        #Find corresponding labels, create 'y' list
        y = []
        x = []
        
        for j in i:
            j = int(j)
            y.append(label[j])
            x.append(featArr[j].tolist())

        print y, x
        prob  = svm_problem(y, x)
        param = svm_parameter('-t 2 -c 4 -b 1')  # Gamma missing
        
        m = svm_train(prob, param)
        allModels.append(m.get_SV())
        svm_save_model('clus' + `order` + '.model', m)
        order+=1

        return allModels

def computeProb(numIns, clusArr):
    prob = []
    for c in clusArr:
        prob.append(float(len(c))/numIns)
    return prob

def main():
########## Test Case  #######################

    

    NUMPOINTS = 200;
    NUMCLASS = 12;
    POINTSPERCLASS = NUMPOINTS/NUMCLASS

    xmin = 0;
    xmax = 100;
    ymin = 0;
    ymax = 100;

    features = np.array([np.zeros(NUMPOINTS), np.zeros(NUMPOINTS)])
    centers = np.array([np.zeros(NUMCLASS), np.zeros(NUMCLASS)])
    isFull = [np.random.randint(0, 2) for r in xrange(NUMPOINTS)]

    classId = list()
    index = 0
##################################### PREPARE FEATURES ###################################
    for i in range(0, NUMCLASS): 
        classId.extend([i]*POINTSPERCLASS)
        centerx = int(np.random.random()*xmax - xmin)
        centery = int(np.random.random()*ymax - ymin)
        centers[0][i] = centerx
        centers[1][i] = centery

        for j in range(0, POINTSPERCLASS):
            datax = int(np.random.normal(loc = centerx, scale = 3))
            datay = int(np.random.normal(loc = centery, scale = 3))

            features[0][index] = datax
            features[1][index] = datay
            index += 1

    # add the remaning points, from integer division
    remainingPoints = NUMPOINTS - POINTSPERCLASS*NUMCLASS
    if (remainingPoints):
        # select the center randomly
        randc = np.random.randint(0, max(classId))
        classId.extend([randc]*remainingPoints)
        for i in range (NUMPOINTS - POINTSPERCLASS*NUMCLASS):
            datax = int(np.random.normal(loc = centers[0][randc], scale = 3))
            datay = int(np.random.normal(loc = centers[1][randc], scale = 3))
            features[0][index] = datax
            features[1][index] = datay
            index += 1
#######################################################################################

    test = getConstraints(NUMPOINTS, isFull, classId)
    
    k = 13
    kmeans = CKMeans(test,features,k)
    output = kmeans.getCKMeans()

 

    """
    probabilities = computeProb(len(labels),clusters)

    print probabilities
    """
#################################################
    # for all clusters
    clustersToBeTrained = list()
    for clusterId in range(len(output[0])):
        # if class id of any that in cluster of clusterId is any different than the first one
        if any(x for x in range(len(output[0][clusterId])) if classId[int(output[0][clusterId][0])] != classId[int(output[0][clusterId][x])]):
            clustersToBeTrained.append(output[0][clusterId])

    allSV = trainSVM(np.transpose(features), clustersToBeTrained, classId)
    visualise.visualiseAfterClustering(allSV,output,np.transpose(features), classId, isFull, "lol")
    plt.show()
    print allSV
if __name__ == '__main__':
    main()
    #profile.run('print main(); print')



