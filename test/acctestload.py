import pickle
from draw import *

numclass, numfull, numpartial = 50, 75, 75
K = [100]  # :O
# K = [numclass]
N = range(1, numclass)
import numpy as np

C = np.linspace(0, 100, 50, endpoint=False)

folderName = '%s__CFPK_%i_%i_%i_%i' % ('Main-CUDA', numclass, numfull, numpartial, K[0])
trainingpath = '../data/training/' + folderName

accuracy = pickle.load(open(trainingpath + '/' "accuracy.p", "r"))
reject_rate = pickle.load(open(trainingpath + '/' "reject.p", "r"))

draw_n_Acc(accuracy, c=0, k=K[0], isfull=True, delay_rate=reject_rate, path=trainingpath)  # for fixed n and c
draw_n_Acc(accuracy, c=0, k=K[0], isfull=False, delay_rate=reject_rate, path=trainingpath)  # for fixed n and c

draw_N_C_Acc(accuracy, N, C, k=K[0], isfull=True, path=trainingpath)
draw_N_C_Reject_Contour(reject_rate, N, C, k=K[0], isfull=True, path=trainingpath)
draw_N_C_Acc_Contour(accuracy, N, C, k=K[0], isfull=True, path=trainingpath)  # Surface over n and c
draw_N_C_Reject_Contour(reject_rate, N, C, k=K[0], isfull=True, path=trainingpath)
# draw_K_Delay_Acc(accuracy, reject_rate, K=K, C=C, n=1, isfull=True, path=trainingpath)
draw_Reject_Acc([accuracy], [reject_rate], N=[1, 2], k=K[0], isfull=True, labels=['Ck-means'], path=trainingpath)

draw_N_C_Acc(accuracy, N, C, k=K[0], isfull=False, path=trainingpath)
draw_N_C_Reject_Contour(reject_rate, N, C, k=K[0], isfull=False, path=trainingpath)
draw_N_C_Acc_Contour(accuracy, N, C, k=K[0], isfull=False, path=trainingpath)  # Surface over n and c
draw_N_C_Reject_Contour(reject_rate, N, C, k=K[0], isfull=False, path=trainingpath)
# draw_K_Delay_Acc(accuracy, reject_rate, K=K, C=C, n=1, isfull=False, path=trainingpath)
draw_Reject_Acc([accuracy], [reject_rate], N=[1, 2], k=K[0], isfull=False, labels=['Ck-means'], path=trainingpath)