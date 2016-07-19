from svmutil import *

m = svm_load_model('clus0.model')
y = [1,2]
p_label, p_acc, p_val = svm_predict(y,[[3,9],[2,2]], m, '-b 1')


def svmProb(model,features):

    y =  [0] * len(features[0])
    p_label, p_acc, p_val = svm_predict(y, features, model, '-b 1')
    return p_val