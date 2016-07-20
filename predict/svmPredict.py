from svmutil import *

m = svm_load_model('clus0.model')
y = [1,2]
p_label, p_acc, p_val = svm_predict(y,[[3,9],[2,2]], m, '-b 1')


def svmProb(model,features):

    y =  [0] * len(features[0])
<<<<<<< HEAD
    p_label, p_acc, p_val = svm_predict(y, features, m, '-b 1')
    return (p_label, p_val)
=======
    p_label, p_acc, p_val = svm_predict(y, features, model, '-b 1')
    return p_val
>>>>>>> ce972b533edacca39b69fe5d84c92142c18a1412
