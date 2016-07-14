"""
Classifier trainer
Ahmet BAGLAN - Arda ICMEZ 
14.07.2016
"""
from svmutil import *
from numpy.random import rand
import numpy as np
import matplotlib.pyplot as plt

def trainSVM(featArr, clusArr, labArr) :

    label = labArr.tolist()
    order = 0
    for i in clusArr:
        #Find corresponding labels, create 'y' list
        y = []
        x = []
        for j in i:
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
########## Test Case  ########################
    cl1 = np.array([0, 1, 2,4,6])
    cl2 = np.array([3,5,7])
    clusters = [cl1,cl2]
    features = np.array([[1, 2], [2, 4], [5, 3], [1, 1], [3, 9], [2, 5], [8, 3], [3, 2]])
    labels = np.array([1,3,2,3,2,1,2,2])

    trainSVM(features, clusters,labels)

########## Calculate probabilities  ########################

    probabilities = computeProb(len(labels),clusters)

    print probabilities

########## Do plotting  ########################

    x = features.tolist()
    y = labels.tolist()

    scale = 80
    color = 'purple'
    for j in range(0,len(y)):
        if(y[j] == 1):
            color = 'red'
        elif(y[j] == 2):
            color = 'blue'
        else:
            color = 'green'
        
        plt.scatter(x[j][0], x[j][1], c=color, s=scale, label=color,
            alpha=0.5, edgecolors='black') 

    plt.grid(True)
    plt.show()

#################################################

   

    
if __name__ == '__main__':
    main()
    #profile.run('print main(); print')



