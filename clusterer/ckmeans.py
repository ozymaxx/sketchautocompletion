"""
CKMeans class
Amir H - Arda I
15.07.2016
"""
import numpy as np
import getConstraints
import sys

class CKMeans:
    
    MUST_LINK = 1
    CANNOT_LINK = -1
    
    def __init__(self, consArr, featArr, k):

        self.consArr = consArr
        self.featArr = featArr
        self.k = k
        self.clusterList = []
    
    def initCluster(self):
        #method to initialize the cluster
        
        for i in range(0,k):
            cluster = np.array([])
            center = self.featArr[i]
            self.clusterList.append((cluster,center))
    
    def violateConstraints(self, data, cluster):
        # method to check if the instance violates the constraints (part of CK-Means)
        # data : id of the instance
        # cluster : current cluster in which we're checking the condition
        
        for i in self.consArr[data]:
            if(i == MUST_LINK):
                if i not in cluster:
                    return true
            elif(i == CANNOT_LINK):
                if i in cluster:
                    return true
        return false
    
    def CKMeans(self) :
        # method to apply CK_Means
        
        # Assign each instance of feature matrix to a cluster
        for i, line in enumerate(self.featArr):
            # i : id of the instance
            # line : points of that instance
            
            availClus = []
            for j in self.clusterList:
                # j : Tuple (Cluster, Center of the cluster)
                
                constraint = self.violateConstraints(i, j[0])
                if not constraint:
                    availClus.append(j)
                    
            if not availClus:
                print "ERROR : No available clusters found"
                
            # Find the closest cluster
            minDist = sys.maxint
            clusNum = 0
            counter = 0
            
            for clus in availClus:
                # clus : Tuple (Cluster, Center of the cluster)
                
                dist = np.linalg.norm(line - clus[1])
                if dist <= minDist:
                    minDist = dist
                    clusNum =counter
                counter+=1
            
            # Assign the instance to the cluster
            self.clusterList[clusNum][0].append(line)
        
        # Find new centers of each cluster    
        for clus in self.clusterList:
            # clus : Tuple (Cluster, Center of the cluster)
            
            dimX = clus[0].shape(1)
            dimY = clus[0].shape(0)
            
            for i in range(0,dimX):
                coorSum = 0
                coorSum = np.sum(clus[0][:,i])
                coorSum /= dimY
                clus[1][i] = coorSum
        
        # Empty out the assigned instances of clusters
        for i in range(0, len(self.clusterList)):
            self.clusterList[0][i] = np.array([])
            
        """
        TODO : Check the convergence, return cluster list
        """
        
def main():
########## Test Case  ########################


    
#################################################
    CKMeans()
   

    
if __name__ == '__main__':
    main()
    #profile.run('print main(); print')
