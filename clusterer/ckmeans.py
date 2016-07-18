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
            center = self.featArr[i] ## TODO : RANDOM ASSIGNMENT OF CENTERS
            self.centerList.append(center)

        
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
        oldCenters = np.zeros([self.k, len(self.featArr[0]) ])
        print oldCenters
        while iterCounter < 5 :
            print iterCounter, "ITER++"
            
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
                print i, "assigned to", clusNum
                print "current elements of that cluster : ", self.clusterList[clusNum]
                
            for i in self.clusterList:
                print i
            ########################################################################
            
            # Save current cluster centers
            for i in range(0, self.k):
                oldCenters[i] = self.centerList[i]
                print oldCenters[i], "Saving clusters"
               
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

def visualiseBeforeClustering2(features):
    color = 'black'
    for x in features.tolist():
        print x[0], x[1]
        scale = 80
        plt.scatter(x[0], x[1], c=color, s=scale, label=color,
            alpha=0.5, edgecolors='black')
    plt.figure()
    plt.grid(True)
    
def visualiseAfterClustering(out,features):
    colorList = cm.rainbow(np.linspace(0, 1, len(out[0])))
    index = 0
    for cluster in out[0]:
        index+=1
        color = colorList[index-1]
            
        for i in cluster.astype(int):
              x = features.tolist()[i]
              scale = 80

              plt.scatter(x[0], x[1], c=color, s=scale, label=color,
                    alpha=0.5, edgecolors='black')

    plt.grid(True)
    
    
def main():

    classId = [1 , 3 , 2 , 1 , 2 , 5 , 2]
    isFull = [1 , 1 , 0 , 0 , 1 , 0 , 1]
    test = getConstraints(7, isFull, classId);
    print test
<<<<<<< HEAD

=======
>>>>>>> 867c2d3b0404bb5207163874a8f51698ced5adc7
    features = np.array([[3,6,5,1,3,2,8],[2,3,3,1,9,5,3]])
    visualiseBeforeClustering2(np.transpose(features))
    kmeans = CKMeans(test,features,4)
    output = kmeans.getCKMeans()
    print output[0]
    print output[1]
    visualiseAfterClustering(output,np.transpose(features))
    plt.show()

if __name__ == '__main__':
    main()
    #profile.run('print main(); print')
