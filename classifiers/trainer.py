"""
Classifier trainer
Ahmet BAGLAN - Arda ICMEZ 
14.07.2016
"""
from svmutil import *
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

        prob  = svm_problem(y, x)
        param = svm_parameter('-t 2 -c 4 -g 0.125 -b 1')  # Gamma missing
        
        m = svm_train(prob, param)
        svm_save_model('clus' + `order` + '.model', m)
        order+=1

def computeProb(numIns, clusArr):
    prob = []
    for c in clusArr:
        prob.append(float(len(c))/numIns)
    return prob

def plotClassCluster(features,labels,clusArr):
    x = features.tolist()
    y = labels.tolist()

    scale = 80
    color = 'purple'
    numClus = len(clusArr)
    marker = 0
    for j in range(0,len(y)):
        if(y[j] == 1):
            color = 'red'
        elif(y[j] == 2):
            color = 'blue'
        else:
            color = 'green'
        for k in range(numClus):
            if( j in clusArr[k]):
                marker = k

        plt.scatter(x[j][0], x[j][1], c=color, s=scale, label=color,
            alpha=0.5, edgecolors='black', marker = (numClus+1,marker+1))

    plt.grid(True)
    plt.show()



def main():
########## Test Case  ########################
    cl1 = np.array([0, 1, 2,4,6])
    cl2 = np.array([3,5,7,9])
    cl3 = np.array([8,10])
    clusters = [cl1,cl2,cl3]
    features = np.array([[1, 2], [2, 4], [5, 3], [1, 1], [3, 9], [2, 5], [8, 3], [3, 2],[2,2],[3,5],[4,4]])
    labels = np.array([1,3,2,3,2,1,2,2,1,2,3])

    trainSVM(features, clusters,labels)

########## Calculate probabilities  ########################

    probabilities = computeProb(len(labels),clusters)

    print probabilities

########## Do plotting  ########################

    plotClassCluster(features,labels,clusters)

#################################################


if __name__ == '__main__':
    main()
    #profile.run('print main(); print')



