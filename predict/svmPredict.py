from svmutil import *

def svmProb(model,features):

    y =  [0] * len(features[0])
    p_label, p_acc, p_val = svm_predict(y, features, model, '-b 1')
    return p_val


m = svm_load_model('../classifiers/clus0.model')

svmProb(m,[[3,9],[2,2]])

