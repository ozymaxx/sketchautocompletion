import numpy as np
import timeit
import scipy.cluster
import scipy.cluster.vq
import math
from numpy.random import randint
from pycuda import driver, compiler, gpuarray
import pycuda.autoinit

class CuCKMeans():
    def __init__(self,features, k, classId, isFull):
        features = np.asarray(features)

        self.features = features
        self.k = k
        self.classId = np.asarray(classId)
        self.isFull = np.asarray(isFull)
        self.fullIndex = [idx for idx in range(len(isFull)) if isFull[idx]]
        
    def cu_vq(self, obs, clusters):
        kernel_code_template = """
         #include "float.h" // for FLT_MAX
           // the kernel definition
          __device__ void loadVector(float *target, float* source, int dimensions )
          {
              for( int i = 0; i < dimensions; i++ ) target[i] = source[i];
          }
          __global__ void cu_vq(float *g_idata, float *g_centroids,
            int *classId, int *isFull,
            int *cluster, float *min_dist, int numClusters, int numDim, int numPoints) {
            int valindex = blockIdx.x * blockDim.x + threadIdx.x ;
            __shared__ float mean[%(DIMENSIONS)s];
            float minDistance = FLT_MAX;
            int myCentroid = 0;
            if(valindex < numPoints){
              for(int k=0;k<numClusters;k++){
                if(threadIdx.x == 0) loadVector( mean, &g_centroids[k*(numDim)], numDim );
                __syncthreads();
                float distance = 0.0;
                for(int i=0;i<numDim;i++){
                  float increased_distance = (g_idata[valindex+i*(numPoints)] -mean[i]);
                  distance = distance +(increased_distance * increased_distance);
                }
                if(distance<minDistance) {
                  minDistance = distance ;
                  myCentroid = k;
                }
              }
              cluster[valindex]=myCentroid;
              min_dist[valindex]=sqrt(minDistance);
            }
          }
        """

        nclusters = clusters.shape[0]
        nclasses = len(np.unique(self.classId))
        points = obs.shape[0]
        dimensions = obs.shape[1]
        block_size = 512
        blocks = int(math.ceil(float(points) / block_size))
    
        kernel_code = kernel_code_template % {
                          'DIMENSIONS': dimensions}
        import platform
        if '1003' in platform.node() and 'Linux' not in platform.system():
            mod = compiler.SourceModule(kernel_code, options=["-ccbin", "C:/Program Files (x86)/Microsoft Visual Studio 12.0/VC/bin/amd64"])
        else:
            mod = compiler.SourceModule(kernel_code)
    
        dataT = obs.T.astype(np.float32).copy()
        clusters = clusters.astype(np.float32)
        
        classIds = self.classId.astype(np.int32).copy()
        isFulls = self.isFull.astype(np.int32).copy()
        
        cluster = gpuarray.zeros(points, dtype=np.int32)
        min_dist = gpuarray.zeros(points, dtype=np.float32)

        
        kmeans_kernel = mod.get_function('cu_vq')
        kmeans_kernel(driver.In(dataT),
                      driver.In(clusters),
                      driver.In(classIds),
                      driver.In(isFulls),
                      cluster,
                      min_dist,
                      np.int32(nclusters),
                      np.int32(dimensions),
                      np.int32(points),
            grid=(blocks, 1),
            block=(block_size, 1, 1),
        )
    
        return cluster.get(), min_dist.get()
    
    def cu_v2q(self, obs, clusters, obs_code, distort, limits):
        kernel_code_template = """
         #include "float.h" // for FLT_MAX
           // the kernel definition
          __device__ void loadVector(float *target, float* source, int dimensions )
          {
              for( int i = 0; i < dimensions; i++ ) target[i] = source[i];
          }
          __global__ void cu_v2q(float *g_idata, float *g_centroids,
            int *classId, int *isFull,
            int * cluster, float *min_dist, int numClusters, int numDim, int numPoints, int numIter) {
            int valindex = blockIdx.x * blockDim.x + threadIdx.x ;
            __shared__ float mean[%(DIMENSIONS)s];
            float minDistance = FLT_MAX;
            int myCentroid = 0;
            if(valindex < numPoints){
              for(int k=numIter;k<numIter+50;k++){
                if(threadIdx.x == 0) loadVector( mean, &g_centroids[k*(numDim)], numDim );
                __syncthreads();
                float distance = 0.0;
                for(int i=0;i<numDim;i++){
                  float increased_distance = (g_idata[valindex+i*(numPoints)] -mean[i]);
                  distance = distance +(increased_distance * increased_distance);
                }
                if(distance<minDistance) {
                  minDistance = distance ;
                  myCentroid = k;
                }
              }
              if(minDistance < (min_dist[valindex]*min_dist[valindex])){
                cluster[valindex]=myCentroid;
                min_dist[valindex]=sqrt(minDistance);
              }
            }
          }
          __global__ void cu_v2qinit(float *g_idata, float *g_centroids,
            int *classId, int *isFull,
            int * cluster, float *min_dist, int numClusters, int numDim, int numPoints, int numIter) {
            int valindex = blockIdx.x * blockDim.x + threadIdx.x ;
            __shared__ float mean[%(DIMENSIONS)s];
            float minDistance = FLT_MAX;
            int myCentroid = 0;
            if(valindex < numPoints){
              for(int k=numIter;k<numIter+50;k++){
                if(threadIdx.x == 0) loadVector( mean, &g_centroids[k*(numDim)], numDim );
                __syncthreads();
                float distance = 0.0;
                for(int i=0;i<numDim;i++){
                  float increased_distance = (g_idata[valindex+i*(numPoints)] -mean[i]);
                  distance = distance +(increased_distance * increased_distance);
                }
                if(distance<minDistance) {
                  minDistance = distance ;
                  myCentroid = k;
                }
              }
              cluster[valindex]=myCentroid;
              min_dist[valindex]=sqrt(minDistance);
            }
          }
        """
        nclusters = clusters.shape[0]
        nclasses = len(np.unique(self.classId))

        points = obs.shape[0]
        dimensions = obs.shape[1]
        block_size = 512
        blocks = int(math.ceil(float(points) / block_size))
        cluster = None
        min_dist = None
        kernel_code = kernel_code_template % {
            'DIMENSIONS': dimensions}
        mod = compiler.SourceModule(kernel_code)
        
        import sys
        import platform
        if '1003' in platform.node() and 'linux' not in sys.platform:
            mod = compiler.SourceModule(kernel_code, options=["-ccbin", "C:/Program Files (x86)/Microsoft Visual Studio 12.0/VC/bin/amd64"])
        else:
            mod = compiler.SourceModule(kernel_code)
    
        dataT = obs.T.astype(np.float32).copy()
        clusters = clusters.astype(np.float32)
        numIter = limits
        
        classIds = self.classId.astype(np.int32).copy()
        isFulls = self.isFull.astype(np.int32).copy()

        if(obs_code is None):
            cluster = gpuarray.zeros(points, dtype=np.int32)
            min_dist = gpuarray.zeros(points, dtype=np.float32)
            kmeans_kernel = mod.get_function('cu_v2qinit')
            kmeans_kernel(driver.In(dataT),
                          driver.In(clusters),
                          driver.In(classIds),
                          driver.In(isFulls),
                          cluster,
                          min_dist,
                          np.int32(nclusters),
                          np.int32(dimensions),
                          np.int32(points),
                          np.int32(numIter),
                          grid=(blocks, 1),
                          block=(block_size, 1, 1),
                          )
        else:
            cluster = gpuarray.to_gpu(obs_code)
            min_dist = gpuarray.to_gpu(distort)
            kmeans_kernel = mod.get_function('cu_v2q')
            kmeans_kernel(driver.In(dataT),
                          driver.In(clusters),
                          driver.In(classIds),
                          driver.In(isFulls),
                          cluster,
                          min_dist,
                          np.int32(nclusters),
                          np.int32(dimensions),
                          np.int32(points),
                          np.int32(numIter),
                          grid=(blocks, 1),
                          block=(block_size, 1, 1),
                          )

        del classIds
        del isFulls
        return cluster.get(), min_dist.get()

    def cu_av(self, obs, clusters, obs_code, classCluster):
        kernel_code_template = """
           // the kernel definition
          __global__ void cu_av(int *obs_code, int *classCluster,int *classId, int *isFull,
            int numPoints) {
            int valindex = blockIdx.x * blockDim.x + threadIdx.x ;
            if(valindex < numPoints){
              if(isFull[valindex] == 1) {
                obs_code[valindex] = classCluster[classId[valindex]];
                }
            }
          }
        """
        nclusters = clusters.shape[0]
        points = obs.shape[0]
        dimensions = obs.shape[1]
        block_size = 512
        blocks = int(math.ceil(float(points) / block_size))
        kernel_code = kernel_code_template % {
                          'DIMENSIONS': dimensions}

        import platform
        if '1003' in platform.node() and 'Linux' not in platform.system():
            mod = compiler.SourceModule(kernel_code, options=["-ccbin", "C:/Program Files (x86)/Microsoft Visual Studio 12.0/VC/bin/amd64"])
        else:
            mod = compiler.SourceModule(kernel_code)
        
        classIds = self.classId.astype(np.int32).copy()
        isFulls = self.isFull.astype(np.int32).copy()
        obs_Code = obs_code.astype(np.int32).copy()
        obs_Code = gpuarray.to_gpu(obs_Code)
        classClusters = np.asarray(classCluster).astype(np.int32).copy()
        
        cluster = gpuarray.zeros(points, dtype=np.int32)
        
        kmeans_kernel = mod.get_function('cu_av')
        kmeans_kernel(obs_Code,
                      driver.In(classClusters),
                      driver.In(classIds),
                      driver.In(isFulls),
                      np.int32(points),
            grid=(blocks, 1),
            block=(block_size, 1, 1),
        )
        del classIds
        del isFulls
        del classClusters
        return obs_Code.get()

    def _cukmeans(self,features, clusters, thresh=1e-5):
        code_book = np.array(clusters, copy=True)  # My clusters centers
        avg_dist = []
        diff = thresh + 1.
        iterNum = 0
        nc = None
        while diff > thresh and iterNum < 100:
            #print "iteration number : ", iterNum
            print 'Iteration number %i (max %i)' %(iterNum, 100)
            nc = code_book.shape[0]  # nc : number of clusters
            
            #if max 150 clusters, do the fast method
            if nc <=150:
                #compute membership and distances between features and code_book
                print "Apply K Means!"
                obs_code, distort = self.cu_vq(features, code_book)
            else:
                #Compute membership with little tasks
                print "Apply K Means!"
                limits = 0
                obs_code, distort = self.cu_v2q(features, code_book, None, None, limits)
                print limits, "-", limits + 50, "finished"
                limits += 50
                while limits <nc:
                    obs_code, distort = self.cu_v2q(features, code_book, obs_code, distort, limits)
                    print limits, "-", limits + 50, "finished"
                    limits += 50

            # Assign full sketches to their own clusters
            nclusters = clusters.shape[0]
            nclasses = len(np.unique(self.classId))
            voteList = np.zeros(nc*nclasses).astype(np.int32)
            for idx in self.fullIndex:
                featclass = self.classId[idx]
                featvote = obs_code[idx]
                voteList[featclass*nc + featvote] += 1

            voteList = np.split(voteList, nclasses) # VOTE list

            # VOTES DOES NOT SUM UP TO NUMBER OF FEATURES
            print "Get Voting!"
            # Find the most voted cluster for every class
            classClusters = []
            i = 0
            
            for myClass in voteList:
                highestIdx = myClass.argmax()
                failsafe = nc
                while (highestIdx in classClusters and failsafe > 0):
                    myClass[highestIdx] = -1
                    highestIdx = myClass.argmax()
                    failsafe -=1
                classClusters.append(highestIdx)
                i+=1

                if failsafe==0:
                    print "Failsafe actvated"
                    
            # assign every full sketch to that cluster
            print "Reassign full sketches!"
            obs_code = self.cu_av(features, code_book, obs_code, classClusters)
            #print nclasses, nclusters, len(voteList), len(instanceVotes)
            #print len(voteList[0]), len(voteList[1]), sum(voteList[0])
            
            # obs_code is the membership of points
            avg_dist.append(np.mean(distort, axis=-1))
            #recalc code_book as centroids of associated features          
            if(diff > thresh):
                has_members = []
                for i in np.arange(nc):
                    cell_members = np.compress(np.equal(obs_code, i), features, 0)
                    if cell_members.shape[0] > 0:
                        code_book[i] = np.mean(cell_members, 0)
                        has_members.append(i)
                #remove code_books that didn't have any members
                code_book = np.take(code_book, has_members, 0)
            if len(avg_dist) > 1:
                diff = avg_dist[-2] - avg_dist[-1]
            iterNum += 1
        
        print "Doing transfer"
        # Transfer obs_code to clusters
        clusterList = []
        centerList = []
        for clus in range(nc):
            clusterList.append(np.array([], dtype = int))
            centerList.append(code_book[clus])          
        for point in range(len(obs_code)):
            clusterList[obs_code[point]] = np.append(clusterList[obs_code[point]], point)

        
        #return  clusterList, code_book, avg_dist[-1]
        return  clusterList, centerList, avg_dist[-1]

    
    def cukmeans(self, thresh=1e-15):
        ITER = 1
        features = self.features
        if type(self.k) == type(np.array([])):
            guess = self.k
            if guess.size < 1:
                raise ValueError("Asked for 0 cluster ? initial book was %s" % \
                                 guess)
            result = self._cukmeans(features, guess, thresh=thresh)
        else:
            #initialize best distance value to a large value
            best_dist = np.inf
            No = features.shape[0]
            k = self.k
            if k < 1:
                raise ValueError("Asked for 0 cluster ? ")
            for i in range(ITER):
                #the intial code book is randomly selected from observations
                print 'Iteration %i (max %i)'%(i, ITER)
                guess = np.take(features, randint(0, No, k), 0) # randomly select cluster centers
                clusters, centers, dist = self._cukmeans(features, guess, thresh=thresh)
                print i,"th CK Means iteration finished"
                if dist < best_dist:
                    print "BEST DISTANCE FOUND, Changed"
                    best_clusters = clusters
                    best_dist = dist
                    best_centers = centers
            result = best_clusters, best_centers
        return result

################################################

if __name__ == "__main__":

    dimensions = 720
    nclusters = 150

    rounds = 1  # for timeit
    rtol = 0.001  # for the relative error calculation

    i = 400
    
    points = 512 * i
    features = np.random.randn(points, dimensions).astype(np.float32) ## WILL GET THIS FROM MAIN
    classId = np.random.randn(points).astype(np.float32) ## WILL GET THIS FROM MAIN
    isFull = np.random.randn(points).astype(np.float32)  ## WILL GET THIS FROM MAIN
    
    print "points", points, "  dimensions", dimensions, "  nclusters", nclusters, "  rounds", rounds
    
    #clusters = data[:nclusters]
    
    clusterer = CuCKMeans(features,nclusters, classId, isFull)  # FEATURES : N x 720
    #print 'pycuda', timeit.timeit(lambda: clusterer.cukmeans(data, clusters), number=rounds)
    clusters, centers = clusterer.cukmeans()

    print type(clusters),type(clusters[0]),type(centers),type(centers[0])
    print len(clusters),len(centers),len(centers[0])
