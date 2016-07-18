"""
CKMeans class
Amir H - Arda I
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
        #method to initialize the cluster
        
        usedPoints = []
        for i in range(0,self.k):
            self.clusterList.append(np.array([]))
            point = randint(0, self.featArr.shape[0])
            while point in usedPoints:
                point = randint(0, len(self.featArr[0])-1)
            usedPoints.append(point)
            center = copy.copy(self.featArr[point]) ## TODO : RANDOM ASSIGNMENT OF CENTERS
            self.centerList.append(center)
        
    def violateConstraints(self, data, cluster):
        # method to check if the instance violates the constraints (part of CK-Means)
        # data : id of the instance
        # cluster : current cluster in which we're checking the condition

        for i in self.consArr[data]:
            if (i == self.MUST_LINK):
                if i not in cluster:
                    return True
            elif (i == self.CANNOT_LINK):
                if i in cluster:
                    return True

        return False

    def getCKMeans(self) :
        # method to apply CK_Means
        self.initCluster()
        # Counter to limit the number of iterations
        iterCounter = 0

        #Old centers of clusters
        oldCenters = np.zeros([self.k, len(self.featArr[0]) ])
        while iterCounter < 20 :

            #Check for convergence
            difference = 0
            for i in range(0, self.k):
                difference += np.linalg.norm(oldCenters[i] - self.centerList[i])

            if difference == 0:
                break
                #pass
            # Empty out the assigned instances of clusters
            for i in range(0, self.k):
                self.clusterList[i] = np.array([])

            ############ Assign each instance of feature matrix to a cluster #############

            for i, line in enumerate(self.featArr):
                # i : id of the instance
                # line : points of that instance

                availClus = []
                for num,j in enumerate(self.clusterList):
                    # j : A cluster
                    # num : order of the iteration

                    constraint = self.violateConstraints(i, j)
                    if not constraint:
                        availClus.append(num)

                if not availClus:
                    print "ERROR : No available clusters found"
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

            # for i in self.clusterList:
            #     print i
            ########################################################################

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

        return (self.clusterList,self.centerList)


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

def visualiseAfterClustering(out, features, classId, centers):
    def getMarkerList():
        numClass = len(set(classId))
        marker_list = list(mark.MarkerStyle.filled_markers)
        markIndex = 4
        while numClass >= len(marker_list):
            marker_list.append((markIndex,1))
            markIndex+=1

        return marker_list

    centers = centers.astype(int)
    colorList = cm.rainbow(np.linspace(0, 1, len(out[0])))
    index = 0
    marker_list = getMarkerList()
    for cluster in out[0]:
        index+=1
        color = colorList[index-1]
        plt.scatter(centers[0][index-1], centers[1][index-1], c='#000000', s=300, label=color,
                    alpha=0.5, edgecolors='black')
        plt.scatter(out[1][index-1][0], out[1][index-1][1], c='red', s=300, label=color,
                    alpha=0.5, edgecolors='black' )


        for i in cluster.astype(int):
              x = features.tolist()[i]
              scale = 80
              marker = classId[i]
              plt
              plt.scatter(x[0], x[1], c=color, s=scale, label=color,
                    alpha=0.5, edgecolors='black', marker= marker_list[marker])

    plt.grid(True)



def main():
    NUMPOINTS = 200;
    NUMCLASS = 5;
    POINTSPERCLASS = NUMPOINTS/NUMCLASS

    xmin = 0;
    xmax = 100;
    ymin = 0;
    ymax = 100;

    #features = np.array.reshape(21)
    features = np.array([np.zeros(NUMPOINTS), np.zeros(NUMPOINTS)])
    centers = np.array([np.zeros(NUMCLASS), np.zeros(NUMCLASS)])
    isFull = [0]*NUMPOINTS
    classId = list()
    index = 0

    for i in range(0, NUMCLASS): 
        classId.extend([i]*POINTSPERCLASS)
        centerx = int(numpy.random.random()*xmax - xmin)
        centery = int(numpy.random.random()*ymax - ymin)
        centers[0][i] = centerx
        centers[1][i] = centery

        for j in range(0, POINTSPERCLASS):
            datax = int(numpy.random.normal(loc = centerx, scale = 3))
            datay = int(numpy.random.normal(loc = centery, scale = 3))

            features[0][index] = datax
            features[1][index] = datay
            index += 1

    test = getConstraints(NUMPOINTS, isFull, classId);
    kmeans = CKMeans(test,features,NUMCLASS)
    output = kmeans.getCKMeans()
    visualiseAfterClustering(output, np.transpose(features), classId, centers)
    plt.show()

    '''
    classId = [1 , 3 , 2 , 3 , 1 , 5 , 2]
    isFull = [1 , 1 , 0 , 0 , 1 , 0 , 1]
    test = getConstraints(7, isFull, classId);
    features = np.array([[3,6,5,1,3,2,8],[2,3,3,1,9,5,3]])
    kmeans = CKMeans(test,features,3)
    output = kmeans.getCKMeans()
    print output[0]
    print output[1]
    visualiseAfterClustering(output,np.transpose(features))
    plt.show()
    '''
if __name__ == '__main__':
    main()
    #profile.run('print main(); print')
