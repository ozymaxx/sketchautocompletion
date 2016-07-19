"""
Classifier trainer
Ahmet BAGLAN - Arda ICMEZ
14.07.2016
"""

import sys
sys.path.append("../clusterer/")
from svmutil import *
from numpy.random import rand
import numpy as np
import matplotlib.pyplot as plt
from ckmeans import *
from getConstraints import *

def trainSVM(featArr, clusArr, labArr) :

    #label = labArr.tolist()
    label = labArr
    order = 0
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
        param = svm_parameter('-t 2 -c 4')  # Gamma missing

        m = svm_train(prob, param)
        svm_save_model('clus' + `order` + '.model', m)
        order+=1

def computeProb(numIns, clusArr):
    prob = []
    for c in clusArr:
        prob.append(float(len(c))/numIns)
    return prob

def main():
########## Test Case  #######################

if __name__ == '__main__':
    main()
    #profile.run('print main(); print')



