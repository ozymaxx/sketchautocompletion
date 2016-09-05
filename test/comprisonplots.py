import pickle
from draw import *

K = [20]
folderName = 'nicicionWithComplexCKMeans_fulldata_newckmean___14_20'

trainingpath = '../data/training/' + folderName

accuracy = pickle.load(open(trainingpath + '/' "accuracy.p", "r"))
reject_rate = pickle.load(open(trainingpath + '/' "reject.p", "r"))

accuracy_parallel_1 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "accuracy_1.p", "r"))
reject_parallel_1 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "reject_1.p", "r"))

accuracy_parallel_2 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "accuracy_2.p", "r"))
reject_parallel_2 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "reject_2.p", "r"))

accuracy_parallel_3 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "accuracy_3.p", "r"))
reject_parallel_3 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "reject_3.p", "r"))

accuracy_parallel_4 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "accuracy_4.p", "r"))
reject_parallel_4 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "reject_4.p", "r"))

accuracy_parallel_5 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "accuracy_5.p", "r"))
reject_parallel_5 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "reject_5.p", "r"))


draw_Reject_Acc([accuracy,  accuracy_parallel_5],
                [reject_rate, reject_parallel_5],
                N=[1, 2], k=K[0], isfull=False, labels=['Ck-means', 'Parallel_5'], path=trainingpath)