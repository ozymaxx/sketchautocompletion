import pickle
from draw import *
import scipy.io as sio
'''
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

mat_contents = sio.loadmat('accresult__niclcon_matlab.mat')
accmesh = mat_contents['accresult']

mat_contents = sio.loadmat('rejresult__partial_niclcon_matlab.mat')
rejmesh = mat_contents['rejresult']

draw_Reject_Acc([accuracy,  accuracy_parallel_5],  [reject_rate, reject_parallel_5],
                N=[1, 2], k=K[0], isfull=False, labels=['Python', '6/1/4/1/2 - 9/1/6/1/3', 'Matlab'], path=trainingpath, additionalacc = [accmesh],
                 additionalrej=[rejmesh],)
'''
K = [250]
folderName = 'complexCudaCKMeanTest___250_70_80_250'

trainingpath = '../data/training/' + folderName

accuracy = pickle.load(open(trainingpath + '/' "accuracy.p", "r"))
reject_rate = pickle.load(open(trainingpath + '/' "reject.p", "r"))

accuracy_parallel_1 = pickle.load(open('/home/semih/Desktop/ahmet_results/ParalelEatzCentersGroupBy25KMeansGroups250_80_80' + '/' + "accuracy.p", "r"))
reject_parallel_1 = pickle.load(open('/home/semih/Desktop/ahmet_results/ParalelEatzCentersGroupBy25KMeansGroups250_80_80' + '/' + "reject.p", "r"))

accuracy_parallel_2 = dict()
reject_parallel_2 = dict()
for keys in accuracy_parallel_1:
    accuracy_parallel_2[(250, keys[1],keys[2],keys[3])] = accuracy_parallel_1[keys]
    reject_parallel_2[(250, keys[1],keys[2],keys[3])] = reject_parallel_1[keys]

accuracy_parallel_1 = accuracy_parallel_2
reject_parallel_1 = reject_parallel_2

'''
accuracy_parallel_2 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "accuracy_2.p", "r"))
reject_parallel_2 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "reject_2.p", "r"))

accuracy_parallel_3 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "accuracy_3.p", "r"))
reject_parallel_3 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "reject_3.p", "r"))

accuracy_parallel_4 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "accuracy_4.p", "r"))
reject_parallel_4 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "reject_4.p", "r"))

accuracy_parallel_5 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "accuracy_5.p", "r"))
reject_parallel_5 = pickle.load(open('/home/semih/Desktop/ahmet_results' + '/' + "reject_5.p", "r"))


mat_contents = sio.loadmat('accresult__niclcon_matlab.mat')
accmesh = mat_contents['accresult']

mat_contents = sio.loadmat('rejresult__partial_niclcon_matlab.mat')
rejmesh = mat_contents['rejresult']
'''

draw_Reject_Acc([accuracy,  accuracy_parallel_1],  [reject_rate, reject_parallel_1],
                N=[1, 2], k=K[0], isfull=False, labels=['Python', '6/1/4/1/2 - 9/1/6/1/3', 'Matlab'], path=trainingpath)