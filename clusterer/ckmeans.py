"""
CKMeans class
Amir H - Arda I
15.07.2016
"""
import numpy as np
from getConstraints import *
import sys

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
        
        for i in range(0,self.k):
            self.clusterList.append(np.array([]))
            center = self.featArr[i]
            self.centerList.append(center)
        print self.clusterList[0], "----", self.clusterList[1]
        print self.centerList[0], "----", self.centerList[1]
        
    def violateConstraints(self, data, cluster):
        # method to check if the instance violates the constraints (part of CK-Means)
        # data : id of the instance
        # cluster : current cluster in which we're checking the condition
        
        for i in self.consArr[data]:
            if(i == self.MUST_LINK):
                if i not in cluster:
                    return True
            elif(i == self.CANNOT_LINK):
                if i in cluster:
                    return True
        return False
    
    def getCKMeans(self) :
        # method to apply CK_Means
        self.initCluster()
        # Counter to limit the number of iterations
        iterCounter = 0
        
        #Old centers of clusters
        oldCenters = np.zeros(self.k)
        
        while iterCounter < 1 :
            print iterCounter, "ITER++"
            
            #Check for convergence
            difference = 0
            for i in range(0, self.k):
                difference += np.linalg.norm(oldCenters[i] - self.centerList[i])
                
            if difference == 0:
                break
            
            ############ Assign each instance of feature matrix to a cluster #############
            
            for i, line in enumerate(self.featArr):
                # i : id of the instance
                # line : points of that instance
                print "i is : ", i, "line is : ", line
                
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
                else:
                    print "available clusters :",availClus
                 
                # Find the closest cluster
                minDist = sys.maxint
                clusNum = 0           
                for num in availClus:
                    # num : id of the available cluster
                    dist = np.linalg.norm(line - self.centerList[num])
                    if dist <= minDist:
                        minDist = dist
                        clusNum = num
                print "closest cluster is :", clusNum
                
                # Assign the instance to the cluster
                self.clusterList[clusNum] = np.append(self.clusterList[clusNum], i)
                print i, "assigned to", clusNum, self.clusterList[0],self.clusterList[1]
                
            for i in self.clusterList:
                print i[0],i[1]
            ########################################################################
            """
            # Save current cluster centers
            for i in range(0, self.k):
                #oldCenters[i] = self.clusterList[1][i]
                print self.clusterList[1][i], "Saving clusters"
                
            # Find new centers of each cluster
            dim = self.featArr.shape[1] #720    
            for order in range(0, self.k):
                
                # clus : Tuple (Cluster, Center of the cluster)
                clus = self.clusterList[order]
                print clus[0],clus[1]
                clusLength = len(clus[0])
                print clusLength,"Cluster uzunlugu"
                
                for i in range(0, dim):
                    # i : order that we're in (0...719)
                    
                    coorSum = 0
                    for j in clus[0]:
                        # j : id of the instance
                        coorSum += self.featArr[j][i]      
                    coorSum /= clusLength
                    self.clusterList[1][order][i] = coorSum

            # Empty out the assigned instances of clusters
            for i in range(0, self.k):
                self.clusterList[0][i] = np.array([])
            """
            # Increment the counter
            iterCounter += 1        
        
        return self.clusterList
    
def main():

    classId = [1 , 3 , 2 , 3 , 1 , 5 , 2]
    isFull = [1 , 1 , 0 , 0 , 1 , 0 , 1]
    test = getConstraints(7, isFull, classId);
    features = np.array([[3,6,5,1,3,2,8],[2,3,3,1,9,5,3]])
    kmeans = CKMeans(test,features,2)    
    kmeans.getCKMeans()
if __name__ == '__main__':
    main()
    #profile.run('print main(); print')
