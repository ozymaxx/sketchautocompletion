"""
Classifier trainer
Ahmet BAGLAN - Arda ICMEZ - Semih GUNEL
14.07.2016
"""

import sys
sys.path.append("../clusterer/")
sys.path.append("../predict/")
sys.path.append("../../libsvm-3.21/python/")

from svmutil import *
from numpy.random import rand
import numpy as np
import matplotlib.pyplot as plt
from ckmeans import *
from getConstraints import *
from testCases import *
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
        param = svm_parameter('-s 0 -t 2 -g 0.125 -c 8 -b 1 -q')
        
        m = svm_train(prob, param)
        allModels.append(m.get_SV())
        svm_save_model('clus' + `order` + '.model', m)
        order+=1

    return allModels

def computeProb(out):
    prob = []
    total =  0

    for c in out[0]:
        total += len(c)

    for cluster in out[0]:
        prob.append(len(cluster)/float(total))
    return prob

def getHeterogenous(output,classId):
    clustersToBeTrained = list()
    for clusterId in range(len(output[0])):
        # if class id of any that in cluster of clusterId is any different than the first one
        if any(x for x in range(len(output[0][clusterId])) if classId[int(output[0][clusterId][0])] != classId[int(output[0][clusterId][x])]):
            clustersToBeTrained.append(output[0][clusterId])
    return  clustersToBeTrained


def main():
########## Test Case  #######################

    

    NUMPOINTS = 200;
    NUMCLASS = 12;
    POINTSPERCLASS = NUMPOINTS/NUMCLASS
    k = 13
###################################################################################

    (features, isFull, classId, centers) = getFeatures(NUMPOINTS,NUMCLASS)

    test = getConstraints(NUMPOINTS, isFull, classId)
    kmeans = CKMeans(test,features,k)
    output = kmeans.getCKMeans()


    # print computeProb(output)
#################################################
    # for all clusters

    clustersToBeTrained = getHeterogenous(output,classId)
    allSV = trainSVM(np.transpose(features), clustersToBeTrained, classId)

    visualise.visualiseAfterClustering(allSV,output,np.transpose(features), classId, isFull, centers, "cluster number not defined")
    plt.show()

if __name__ == '__main__':
    main()
    #profile.run('print main(); print')



