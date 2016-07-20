from svmutil import *

def svmProb(model,features):

    m = svm_load_model('../classifiers/' + model)
    y =  [0] * len(features[0])
    p_label, p_acc, p_val = svm_predict(y, features, m, '-b 1')
    return (p_label, p_val)
