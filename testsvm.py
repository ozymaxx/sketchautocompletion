from svmutil import *

def trainSVM(featArr, clusArr, labArr) :
    
    label = labArr.tolist()
    order = 0
    for i in len(clusArr):
        
        #Find corresponding labels, create 'y' list
        y = []
        x = []
        for j in i:
            y.append(label[j])
            x.append(featArr[j])
        
        print y
        print x
        """
        prob  = svm_problem(y, x, isKernel=True)
        param = svm_parameter('-t 2 -c 4')  # Gamma missing
        
        m = svm_train(prob, param)
        svm_save_model('clus' + `order` + '.model', m)
        order+=1
        """
def main():
    pass
    
if __name__ == '__main__':
    main()
    #profile.run('print main(); print')