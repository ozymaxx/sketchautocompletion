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
import threading

class workerThread (threading.Thread):
    def __init__(self, threadID, name , numThread, consArr):
        raise DeprecationWarning
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.numThread = numThread
        self.consArr = consArr

    def violateConstraints(self, data, cluster):
        """method to check if the instance violates the constraints (part of CK-Means)
        data : id of the instance
        cluster : current cluster in which we're checking the condition
        return any(bool(self.consArr[data][i] == self.MUST_LINK) != bool(i in cluster) for i in range(0,data))"""
        for i in range(0, data):
            if (self.consArr[data][i] == 1):
                if i not in cluster:
                    return True
            elif (self.consArr[data][i] == -1):
                if i in cluster:
                    return True
        return False
    
    def assignCluster(self):
        global threadFlag
        global jobCount
        global featThread
        global clusListThread
        global centListThread
        i = self.threadID
        while ( i < featThread.shape[0] ):
            # i : id of the instance
            # line : points of that instance

            availClus = []
            line = featThread[i]
            for num,j in enumerate(clusListThread):
                # j : A cluster
                # num : order of the iteration
                
                #constraint = self.violateConstraints(i , j)
                constraint = False
                if not constraint:
                    availClus.append(num)

            if not availClus:
                print "ERROR : No available clusters found for",i ,"th instance"
                i += self.numThread
                continue

            # Find the closest cluster
            minDist = sys.maxint
            clusNum = 0
            for num in availClus:
            # num : id of the available cluster
                dist = np.linalg.norm(line - centListThread[num])
                if dist <= minDist:
                    minDist = dist
                    clusNum = num

            # Assign the instance to the cluster
            clusListThread[clusNum] = np.append(clusListThread[clusNum], i)
            i += self.numThread
            
    def findCenter(self):
        global threadFlag
        global jobCount
        global featThread
        global clusListThread
        global centListThread
        # Find new centers of each cluster
        dim = featThread.shape[1] #720
        order = self.threadID
        while order < len(clusListThread):

            clus = copy.copy(clusListThread[order])
            clusLength = len(clus)

            for i in range(0, dim):
                # i : order that we're in (0...719)

                coorSum = 0
                for j in clus:
                    # j : id of the instance
                    coorSum += copy.copy(featThread[j][i])
                if coorSum != 0:
                    coorSum /= clusLength
                    centListThread[order][i] = coorSum
            order += self.numThread
    
    def run(self):
        global threadFlag
        global jobCount
        global featThread
        global clusListThread
        global centListThread
        print "thread",self.threadID,"start"
        selfFlag = 1
        while True:
            if (threadFlag == 10 and selfFlag == 1):
                self.assignCluster()
                jobCount+=1
                selfFlag = 0
            elif (threadFlag == 20 and selfFlag == 0):
                self.findCenter()
                jobCount -= 1
                selfFlag = 1
            elif (threadFlag == -1):
                return

class masterThread (threading.Thread):
     
    def __init__(self, name , numThread, k):
        threading.Thread.__init__(self)
        self.name = name
        self.numThread = numThread
        self.k = k
        
    def run(self):
        global threadFlag
        global jobCount
        global featThread
        global clusListThread
        global centListThread
        
        # Counter to limit the number of iterations
        iterCounter = 0
        #Old centers of clusters
        oldCenters = np.zeros([self.k, len(featThread[0])])
        while iterCounter < 20:
            print 'Constrained k-means iteation: ' + str(iterCounter+1) + ' (max 20)'
            
            #Check for convergence ---- MAYBE BOTTLENECK?
            difference = 0
            for i in range(0, self.k):
                difference += np.linalg.norm(oldCenters[i] - centListThread[i])
                #print np.linalg.norm(oldCenters[i] - centListThread[i])
            # checking whether a fp is zero?
            if difference < pow(10, -10):
                threadFlag = -1
                print "Master : Found convergence"
                break
            
            ## Thread job id : 10
            threadFlag = 10
            
            
            # wait until all threads are finished
            while(jobCount != self.numThread):pass

            # Save current cluster centers ---- MAYBE BOTTLENECK?
            for i in range(0, self.k):
                oldCenters[i] = copy.copy(centListThread[i])
                # print oldCenters[i], "Saving clusters"
                
            ## Thread job id : 20
            threadFlag = 20
            
            # wait until all threads are finished
            while(jobCount):pass
            
            for kls in centListThread:
                print kls
            
            # Empty out the assigned instances of clusters ---- MAYBE BOTTLENECK?
            for i in range(0, self.k):
                clusListThread[i] = np.array([], dtype=int)
            
            # Increment the counter
            iterCounter += 1

        # Signal all threads to exit 
        threadFlag = -1
        
class CKMeans:
    def __init__(self, consArr, featArr, k, numThread):
        self.consArr = consArr
        self.featArr = np.transpose(featArr)
        self.k = k
        self.clusterList = []
        self.centerList = []
        self.MUST_LINK = 1
        self.CANNOT_LINK = -1
        self.numThread = numThread
        
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

    def runThreads(self):
        
        #Initialize clusters
        self.initCluster()
        
        #Set shared variables between threads 
        global featThread
        global clusListThread
        global centListThread
        global threadFlag
        global jobCount
        featThread = self.featArr
        clusListThread = self.clusterList
        centListThread = self.centerList
        threadFlag = 0
        jobCount = 0
        
        #Set number of threads
        myThreads = []
        
        #Start threads
        master = masterThread("Master Thread",self.numThread, self.k)
        master.start()
        myThreads.append(master)
        for i in range(0,self.numThread):
            thread = workerThread(i,"Thread0"+`i`,self.numThread, self.consArr)
            thread.start()
            myThreads.append(thread)
            
        # Synchronize threads
        for t in myThreads:
            t.join()
            
        return (clusListThread, centListThread)
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


def main():
    numpoint = 2000
    numclass = 10
    numthread = 2
    features,isFull,classId,centers = getFeatures(numpoint,numclass)
    constArray = getConstraints(numpoint, isFull, classId)
    l = CKMeans(constArray,features,numclass,numthread)
    l.runThreads()
if __name__ == "__main__": main()