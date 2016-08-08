import numpy as np
import timeit
import scipy.cluster
import scipy.cluster.vq
import math
from numpy.random import randint
from pycuda import driver, compiler, gpuarray
import pycuda.autoinit

class CuCKMeans():
    def __init__(self, consArr,features,k):
        self.consArr = consArr
        self.features = features
        self.k = k
        
    def cu_vq(self,obs, clusters):
        kernel_code_template = """
         #include "float.h" // for FLT_MAX
           // the kernel definition
          __device__ void loadVector(float *target, float* source, int dimensions )
          {
              for( int i = 0; i < dimensions; i++ ) target[i] = source[i];
          }
          __global__ void cu_vq(float *g_idata, float *g_centroids,
            int * cluster, float *min_dist, int numClusters, int numDim, int numPoints) {
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
        points = obs.shape[0]
        dimensions = obs.shape[1]
        block_size = 512
        blocks = int(math.ceil(float(points) / block_size))
    
        kernel_code = kernel_code_template % {
                          'DIMENSIONS': dimensions}
        mod = compiler.SourceModule(kernel_code)
    
        dataT = obs.T.astype(np.float32).copy()
        clusters = clusters.astype(np.float32)
    
        cluster = gpuarray.zeros(points, dtype=np.int32)
        min_dist = gpuarray.zeros(points, dtype=np.float32)
    
        kmeans_kernel = mod.get_function('cu_vq')
        kmeans_kernel(driver.In(dataT),
                      driver.In(clusters),
                      cluster,
                      min_dist,
                      np.int32(nclusters),
                      np.int32(dimensions),
                      np.int32(points),
            grid=(blocks, 1),
            block=(block_size, 1, 1),
        )
    
        return cluster.get(), min_dist.get()
    
    
    def _cukmeans(self,features, clusters, thresh=1e-5):

        code_book = np.array(clusters, copy=True)  # My clusters centers
        avg_dist = []
        diff = thresh + 1.
        iterNum = 0
        nc = None
        while diff > thresh and iterNum < 20:
            print "iteration number : ", iterNum
            nc = code_book.shape[0]  # nc : number of clusters
            #compute membership and distances between features and code_book
            
            obs_code, distort = self.cu_vq(features, code_book)
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

    
    def cukmeans(self, thresh=1e-5):
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
                guess = np.take(features, randint(0, No, k), 0) # randomly select cluster centers
                clusters, centers, dist = self._cukmeans(features, guess, thresh=thresh)
                if dist < best_dist:
                    best_clusters = clusters
                    best_dist = dist
                    best_centers = centers
            result = best_clusters, best_centers
        return result

################################################

if __name__ == "__main__":

    dimensions = 720
    nclusters = 20

    rounds = 1  # for timeit
    rtol = 0.001  # for the relative error calculation

    i = 10
    
    points = 512 * i
    features = np.random.randn(points, dimensions).astype(np.float32)
    print "points", points, "  dimensions", dimensions, "  nclusters", nclusters, "  rounds", rounds
    
    #clusters = data[:nclusters]
    
    clusterer = CuCKMeans(None,features,nclusters)  # FEATURES : N x 720
    #print 'pycuda', timeit.timeit(lambda: clusterer.cukmeans(data, clusters), number=rounds)
    clusters, centers = clusterer.cukmeans()
    
    print type(clusters),type(clusters[0]),type(centers),type(centers[0])
    print len(clusters),len(centers),len(centers[0])
