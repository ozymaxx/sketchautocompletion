from svmutil import *

def svmProb(model,features):

    y =  [0] * len(features[0])
<<<<<<< HEAD
    p_label, p_acc, p_val = svm_predict(y, features, m, '-b 1')
    return (p_label, p_val)
=======
    p_label, p_acc, p_val = svm_predict(y, features, model, '-b 1')
    return p_val
<<<<<<< HEAD
>>>>>>> ce972b533edacca39b69fe5d84c92142c18a1412
=======


m = svm_load_model('../classifiers/clus0.model')

svmProb(m,[[3,9],[2,2]])

>>>>>>> 115f8fa2e749a4132afec1b7fd6a2a96c52b5fdd
