"""
CKMeans class
Arda I - Semih G - Ahmet B
15.07.2016
"""

import numpy as np
from getConstraints import *
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from random import randint
import copy
import matplotlib.markers as mark

class CKMeans:
    def __init__(self, consArr, featArr, k):
        self.consArr = consArr
        self.featArr = np.transpose(featArr)
        self.k = k
        self.clusterList = []
        self.centerList = []
        self.MUST_LINK = 1
        self.CANNOT_LINK = -1

    def initCluster(self):
        """method to initialize the clusters"""

        usedPoints = []
        for i in range(0,self.k):
            self.clusterList.append(np.array([], dtype = int))
            
            # Select unique cluster centers randomly 
            point = randint(0, self.featArr.shape[0]-1)
            while point in usedPoints:
                point = randint(0, self.featArr.shape[0]-1)
            usedPoints.append(point)

            center = copy.copy(self.featArr[point])
            self.centerList.append(center)
        
    def violateConstraints(self, data, cluster):
        """method to check if the instance violates the constraints (part of CK-Means)
        data : id of the instance
        cluster : current cluster in which we're checking the condition
        return any(bool(self.consArr[data][i] == self.MUST_LINK) != bool(i in cluster) for i in range(0,data))"""
        for i in range(0, data):
            if (self.consArr[data][i] == self.MUST_LINK):
                if i not in cluster:
                    return True
            elif (self.consArr[data][i] == self.CANNOT_LINK):
                if i in cluster:
                    return True
        return False

    def getCKMeans(self):
        """method to apply CK_Means"""
        self.initCluster()
        # Counter to limit the number of iterations
        iterCounter = 0

        #Old centers of clusters
        oldCenters = np.zeros([self.k, len(self.featArr[0])])
        while iterCounter < 20:
            print 'Constrained k-means iteation: ' + str(iterCounter+1) + ' (max 20)'
            #Check for convergence
            difference = 0
            for i in range(0, self.k):
                difference += np.linalg.norm(oldCenters[i] - self.centerList[i])

            # checking whether a fp is zero?
            if difference < pow(10, -10):
                break

            # Empty out the assigned instances of clusters
            for i in range(0, self.k):
                self.clusterList[i] = np.array([], dtype=int)

            ############ Assign each instance of feature matrix to a cluster #############

            for i, line in enumerate(self.featArr):
                # i : id of the instance
                # line : points of that instance

                availClus = []
                for num,j in enumerate(self.clusterList):
                    # j : A cluster
                    # num : order of the iteration

                    constraint = self.violateConstraints(i, j)
                    #constraint = False
                    if not constraint:
                        availClus.append(num)

                if not availClus:
                    print "ERROR : No available clusters found for",i ,"th instance"
                    continue

                # Find the closest cluster
                minDist = sys.maxint
                clusNum = 0
                for num in availClus:
                    # num : id of the available cluster
                    dist = np.linalg.norm(line - self.centerList[num])
                    if dist <= minDist:
                        minDist = dist
                        clusNum = num

                # Assign the instance to the cluster
                self.clusterList[clusNum] = np.append(self.clusterList[clusNum], i)

            # Save current cluster centers
            for i in range(0, self.k):
                oldCenters[i] = self.centerList[i]
                # print oldCenters[i], "Saving clusters"

            # Find new centers of each cluster
            dim = self.featArr.shape[1] #720
            for order in range(0, self.k):

                clus = self.clusterList[order]
                clusLength = len(clus)

                for i in range(0, dim):
                    # i : order that we're in (0...719)

                    coorSum = 0
                    for j in clus:
                        # j : id of the instance
                        coorSum += self.featArr[j][i]
                    if coorSum != 0:
                        coorSum /= clusLength
                        self.centerList[order][i] = coorSum

            # Increment the counter
            iterCounter += 1

        return (self.clusterList, self.centerList)

def visualiseBeforeClustering(out,features):
    color = 'black'
    for cluster in out[0]:
        for i in cluster.astype(int):
              x = features.tolist()[i]
              scale = 80
              plt.scatter(x[0], x[1], c=color, s=scale, label=color,
                    alpha=0.5, edgecolors='black')
    plt.figure()
    plt.grid(True)

def visualiseAfterClustering(out, features, classId, centers, isFull, title):
    def getMarkerList():
        numClass = len(set(classId))
        marker_list = list(mark.MarkerStyle.filled_markers)
        markIndex = 4
        while numClass >= len(marker_list):
            marker_list.append((markIndex,1))
            markIndex+=1

        return marker_list

    fig1 = plt.figure()
    fig1.canvas.set_window_title(str(title))
    ax1 = fig1.add_subplot(111)

    fig2 = plt.figure()
    fig2.canvas.set_window_title("K="+str(title) + " Full Sketch")
    ax2 = fig2.add_subplot(111)

    centers = centers.astype(int)
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
                    alpha=0.5, edgecolors='black' )

        for i in cluster.astype(int):
              x = features.tolist()[i]
              scale = 80
              marker = classId[i]

              ax1.scatter(x[0], x[1], c=color, s=scale, label=color,
                    alpha=0.5, edgecolors='black', marker= marker_list[marker])

              if(isFull[i] == 1):
                count += 1
                ax2.scatter(x[0], x[1], c=color, s=scale, label=color,
                    alpha=0.5, edgecolors='black', marker= marker_list[marker])

    print count
    plt.grid(True)
    ax1.grid(True)

def getFeatures(NUMPOINTS, NUMCLASS):  #k is already the number of clusters
    POINTSPERCLASS = NUMPOINTS / NUMCLASS

    xmin = 0;
    xmax = 100;
    ymin = 0;
    ymax = 100;

    features = np.array([np.zeros(NUMPOINTS), np.zeros(NUMPOINTS)])
    centers = np.array([np.zeros(NUMCLASS), np.zeros(NUMCLASS)])
    isFull = [np.random.randint(0, 2) for r in xrange(NUMPOINTS)]

    classId = list()
    index = 0
##################################### PREPARE FEATURES ###################################
    for i in range(0, NUMCLASS):
        classId.extend([i]*POINTSPERCLASS)
        centerx = int(np.random.random()*(xmax - xmin) + xmin)
        centery = int(np.random.random()*(ymax - ymin) + ymin)
        centers[0][i] = centerx
        centers[1][i] = centery

        for j in range(0, POINTSPERCLASS):
            datax = int(np.random.normal(loc = centerx, scale = 3))
            datay = int(np.random.normal(loc = centery, scale = 3))

            features[0][index] = datax
            features[1][index] = datay
            index += 1

    # add the remaning points, from integer division
    remainingPoints = NUMPOINTS - POINTSPERCLASS * NUMCLASS
    if (remainingPoints):
        # select the center randomly
        randc = np.random.randint(0, max(classId))
        classId.extend([randc]*remainingPoints)
        for i in range (NUMPOINTS - POINTSPERCLASS*NUMCLASS):
            datax = int(np.random.normal(loc = centers[0][randc], scale = 3))
            datay = int(np.random.normal(loc = centers[1][randc], scale = 3))
            features[0][index] = datax
            features[1][index] = datay
            index += 1

    return  (features,isFull,classId,centers)


