from svmutil import *

def svmProb(model,instance):

    y = [0]
    p_label, p_acc, p_val = svm_predict(y, instance, model, '-b 1')
    return (p_label, p_val)
