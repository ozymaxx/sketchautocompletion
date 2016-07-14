from svmutil import *
import numpy as np

def trainSVM(featArr, clusArr, labArr) :
    
    label = labArr.tolist()
    order = 0
    for i in clusArr:
        #Find corresponding labels, create 'y' list
        y = []
        x = []
        for j in i:
            y.append(label[j-1])
            x.append(featArr[j-1].tolist())

        print y, x
        prob  = svm_problem(y, x)
        param = svm_parameter('-t 2 -c 4')  # Gamma missing
        
        m = svm_train(prob, param)
        svm_save_model('clus' + `order` + '.model', m)
        order+=1

def main():

    cl1 = np.array([1, 2, 3,5,7])
    cl2 = np.array([4,6,8])
    clusters = [cl1,cl2]
    features = np.array([[1, 2], [3, 9], [3, 9], [3, 9], [3, 9], [3, 9], [3, 9], [3, 9]])
    labels = np.array([1,3,2,3,2,1,2,2])

    trainSVM(features, clusters,labels)

    
if __name__ == '__main__':
    main()
    #profile.run('print main(); print')