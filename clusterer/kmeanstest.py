import numpy as np
from getConstraints import *
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from random import randint
import copy
import matplotlib.markers as mark
#from cudackmeans import *
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append('../test/')
sys.path.append('../data/')
from extractor import *
from FileIO import *
from Predictor import *
from visualise import *
from fastCKMeans import *
from scipykmeans import *
from scipyCKMeans import *
from cudackmeans import *
from complexcudackmeans import *
import random
def visualiseAfterClustering(out, features, classId, centers, isFull, title, sv):
    def getMarkerList():
        numClass = len(set(classId))
        marker_list = list(mark.MarkerStyle.filled_markers)
        markIndex = 4
        while numClass >= len(marker_list):
            marker_list.append((markIndex,1))
            markIndex+=1

        return marker_list
    clisEdge = ['black', 'yellow', 'red', 'blue']
    colorList = cm.rainbow(np.linspace(0, 1, len(out[0])))
    features = np.transpose(features)
    fig1 = plt.figure()
    fig1.canvas.set_window_title(str(title))
    ax1 = fig1.add_subplot(111)

    fig2 = plt.figure()
    fig2.canvas.set_window_title("K="+str(title) + " Full Sketch")
    ax2 = fig2.add_subplot(111)

    centers = np.asarray(centers).astype(int)
    colorList = cm.rainbow(np.linspace(0, 1, len(out[0])))
    #print len(centers[0])
    for i in range(len(centers[0])):
         ax1.scatter(centers[0][i], centers[1][i], c='#000000', s=300,
                    alpha=0.5, edgecolors='black')
    index = 0
    marker_list = getMarkerList()

    count = 0
    for cluster in out[0]:
        index+=1
        color = colorList[index-1]
        ax1.scatter(out[1][index-1][0], out[1][index-1][1], c='red', s=300, label=color,
                    alpha=0.5, edgecolors='black')

        color = colorList[index - 1]
        edgecolor = colorList[random.randint(0, len(clisEdge)) - 1]

        for i in cluster:
            x = features.tolist()[i]
            scale = 80
            marker = classId[i]

            ax1.scatter(x[0], x[1], c=color, s=scale, label=color,
                alpha=0.5, edgecolors=edgecolor, marker= marker_list[marker],linewidth='2')

            if(isFull[i] == 1):
                count += 1
                ax2.scatter(x[0], x[1], c=color, s=scale, label=color,
                    alpha=0.5, edgecolors='black', marker= marker_list[marker],linewidth='2')

    if sv:
        sv_x, sv_y = [], []
        for point in sv:
            try:
                sv_x.append(point[1])
                sv_y.append(point[2])
            except:
                pass

        plt.scatter(sv_x, sv_y, c='black', s=100, label=color,
                    alpha=0.5, edgecolors='black', marker='+')

        ax1.scatter(sv_x, sv_y, c='black', s=100, label=color,
                    alpha=0.5, edgecolors='black', marker='+')

        ax2.scatter(sv_x, sv_y, c='black', s=100, label=color,
                    alpha=0.5, edgecolors='black', marker='+')

    plt.grid(True)
    ax1.grid(True)
    plt.show(block=True)

def main():
    numpoint = 200
    numclass = 10
    features,isFull,classId,centers = getFeatures(numpoint, numclass)
    constArray = getConstraints(numpoint, isFull, classId)
    '''
    l = CKMeans(constArray, features, 10)
    kmeansoutput = l.getCKMeans()

    trainerkmeans = Trainer(kmeansoutput, classId, np.transpose(features))
    heteClstrFeatureId, heteClstrId = trainerkmeans.getHeterogenous()

    svmkmeans = trainerkmeans.trainSVM(heteClstrFeatureId, None)

    predictor = Predictor(kmeansoutput, classId, None, svm=svmkmeans)
    svkmeans = predictor.getSV()

    visualiseAfterClustering( kmeansoutput, features, classId, centers, isFull, 'k-means', sv=svkmeans)
    '''

    clusterer = complexCudaCKMeans(np.transpose(features), numclass, classId, isFull, stepweight= 0.05, maxweight=1)
    kmeansoutput = clusterer.cukmeans()

    trainer = Trainer(kmeansoutput, classId, np.transpose(features))
    heteClstrFeatureId, heteClstrId = trainer.getHeterogenousClusterId()

    svmkmeans = trainer.trainSVM(heteClstrFeatureId, None)

    predictor = Predictor(kmeansoutput, classId, None, svm=svmkmeans)
    svkmeans = predictor.getSupportVectors()

    visualiseAfterClustering(kmeansoutput, features, classId, centers, isFull, 'cudackmeans', sv=svkmeans)

    ckmeans = scipyCKMeans(np.transpose(features), isFull, classId, k=numclass, maxiter=20, thres=10 ** -10)
    kmeansoutput = ckmeans.getCKMeans()

    trainer = Trainer(kmeansoutput, classId, np.transpose(features))
    heteClstrFeatureId, heteClstrId = trainer.getHeterogenousClusterId()

    svm = trainer.trainSVM(heteClstrFeatureId, None)

    predictor = Predictor(kmeansoutput, classId, None, svm=svm)
    sv = predictor.getSupportVectors()

    visualiseAfterClustering(kmeansoutput, features, classId, centers, isFull, 'ck-means', sv=sv)

    clusterclassid = [[classId[featureidx] for featureidx in cluster if isFull[featureidx]] for cluster in kmeansoutput[0]]
    print any([(len(set(cluster)) != 1) for cluster in clusterclassid])

if __name__ == "__main__": main()