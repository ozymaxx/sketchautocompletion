from svmutil import *

m = svm_load_model('clus0.model')

y = [1,2]
p_label, p_acc, p_val = svm_predict(y,[[3,9],[2,2]], m, '-b 1')

print p_label,"label"
print "accuracy ",p_acc
print "val ", p_val