import pickle
from draw import *

"""
Loads test results - with pickle - and generates plots, without a
undergoing a testing process again.

Do that if you want different plots than templates, or in order
to regenerate the plots with different parameters.
"""

numclass, numfull, numpartial = 13, 80, 80
K = [20]  # :O
fullness = range(1,11)
# K = [numclass]
N = range(1, numclass+1)
import numpy as np

C = np.linspace(0, 100, 51, endpoint=True)
C = [int(c) for c in C]

#folderName = '%s___%i_%i' % ('nicicionWithCuda_fulldata', 10, 20)
folderName = 'nicicionWithComplexCKMeans_fulldata_newprior___14_20'

trainingpath = '../data/training/' + folderName

accuracy = pickle.load(open(trainingpath + '/' "accuracy.p", "r"))
reject_rate = pickle.load(open(trainingpath + '/' "reject.p", "r"))
#accuracy_fullness = pickle.load(open(trainingpath + '/' "accuracy_fullness.p", "r"))

#draw_Completeness_Accuracy(accuracy_fullness, fullness, k=K[0], n=1, C=np.linspace(0, 80, 5), isfull=False,
#                           path=trainingpath)

draw_N_C_Acc_Contour_Low(accuracy, N, C, k=K[0], isfull=False, path=trainingpath, title='For Python Implementation Using Niclcon dataset')
draw_N_C_Acc_Contour_Low(accuracy, N, C, k=K[0], isfull=True, path=trainingpath, title='For Python Implementation Using Niclcon dataset')
draw_N_C_Rej_Contour_Low(reject_rate, N, C, k=K[0], isfull=False, path=trainingpath, title='For Python Implementation Using Niclcon dataset')

#draw_n_Acc(accuracy, c=0, k=K[0], isfull=True, reject_rate=reject_rate, path=trainingpath)
#draw_n_Acc(accuracy, c=0, k=K[0], isfull=False, reject_rate=reject_rate, path=trainingpath)

#draw_N_C_Acc_Contour_Low(accuracy, N, C, k=K[0], isfull=False, path=trainingpath)
#draw_N_C_Acc_Contour(accuracy, N, C, k=K[0], isfull=False, path=trainingpath)

#draw_N_C_Acc(accuracy, N, C, k=K[0], isfull=True, path=trainingpath)
#draw_N_C_Reject_Contour(reject_rate, N, C, k=K[0], isfull=True, path=trainingpath
#draw_N_C_Acc_Contour(accuracy, N, C, k=K[0], isfull=True, path=trainingpath)  # Surface over n and c
#draw_N_C_Reject_Contour(reject_rate, N, C, k=K[0], isfull=True, path=trainingpath)
#draw_K_Delay_Acc(accuracy, reject_rate, K=K, C=C, n=1, isfull=True, path=trainingpath)
#draw_Reject_Acc([accuracy], [reject_rate], N=[1, 2], k=K[0], isfull=True, labels=['Ck-means'], path=trainingpath)

#draw_N_C_Acc(accuracy, N, C, k=K[0], isfull=False, path=trainingpath)
#draw_N_C_Reject_Contour(reject_rate, N, C, k=K[0], isfull=False, path=trainingpath)
#draw_N_C_Acc_Contour(accuracy, N, C, k=K[0], isfull=False, path=trainingpath)  # Surface over n and c
#draw_N_C_Reject_Contour(reject_rate, N, C, k=K[0], isfull=False, path=trainingpath)
#draw_K_Delay_Acc(accuracy, reject_rate, K=K, C=C, n=1, isfull=False, path=trainingpath)
#draw_Reject_Acc([accuracy], [reject_rate], N=[1, 2], k=K[0], isfull=False, labels=['Ck-means'], path=trainingpath)

print 'End'
