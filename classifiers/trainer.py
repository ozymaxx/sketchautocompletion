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

    cl1 = np.array([0, 1, 2,4,6])
    cl2 = np.array([3,5,7])
    clusters = [cl1,cl2]
    features = np.array([[1, 2], [3, 9], [3, 9], [3, 9], [3, 9], [3, 9], [3, 9], [3, 9]])
    labels = np.array([1,3,2,3,2,1,2,2])
<<<<<<< Updated upstream
    trainSVM(features, clusters,labels)
=======

    #trainSVM(features, clusters,labels)
    
    for color in ['red', 'green', 'blue']:
        n = 750
        x, y = rand(2, n)
        print x
        scale = 200.0 * rand(n)
        plt.scatter(x, y, c=color, s=scale, label=color,
                    alpha=0.3, edgecolors='none')

    plt.legend()
    plt.grid(True)
    
    plt.show()
>>>>>>> Stashed changes

    probabilities = computeProb(len(labels),clusters)

    print probabilities

    
if __name__ == '__main__':
    main()
    #profile.run('print main(); print')



