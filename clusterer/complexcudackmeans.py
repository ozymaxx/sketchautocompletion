import numpy as np
import timeit
import scipy.cluster
import scipy.cluster.vq
import math
from numpy.random import randint
from pycuda import driver, compiler, gpuarray
import pycuda.autoinit
import operator

class complexCudaCKMeans():
    def __init__(self, features, k, classId, isFull, initweight=0, maxweight=2, stepweight=0.2, maxiter = 20):
        features = np.asarray(features)

        self.numclass = len(set(classId))
        self.features = features
        self.k = k
        self.classId = np.asarray(classId)
        self.isFull = np.asarray(isFull)
        self.fullIndex = [idx for idx in range(len(isFull)) if isFull[idx]]
        self.clusterCenters = [[0]*len(features[0]) for i in range(k)]
        self.clusterFeatures = [[] for i in range(k)]
        self.classvotes = [[0]*self.k for i in range(self.numclass)] # votes of the classes for cluster, row for class
        self.class2cluster = [0]*self.numclass
        self.featureClusterDist = [[0]*self.k for i in range(len(self.features))] # distance of feature to cluster
        self.featureCluster = [0]*len(features)

        self.initweight = initweight
        self.maxweight = maxweight
        self.stepweight = stepweight
        self.currweight = initweight

        self.maxiter = maxiter

        self.initClusters()

    def initClusters(self):
        # randomly select from features
        # self.clusterCenters = copy.copy(random.sample(self.features, self.k))
        print 'Initializing Cluster Centers'
        numFeatureAddedForCluster = [0] * self.k
        # initialize numclass of k cluster to be the center of the full sketches
        if self.numclass <= self.k:
            for fidx in self.fullIndex:
                fclass = self.classId[fidx]
                if fclass < self.k:
                    # add each full sketch to the corresponding cluster, then divide
                    self.clusterCenters[fclass] = map(operator.add,
                                                      self.clusterCenters[fclass],
                                                      self.features[fidx])
                    numFeatureAddedForCluster[fclass] += 1

            for clusterCenterIdx in range(self.numclass):
                self.clusterCenters[clusterCenterIdx] = [cfloat / numFeatureAddedForCluster[clusterCenterIdx] for cfloat
                                                         in self.clusterCenters[clusterCenterIdx]]

        # for the remaining cluster centers, randomly select from the non-selected features
        numClustSelected = self.numclass
        while numClustSelected < self.k:
            featIdx = randint(0, len(self.features))
            if not self.isFull[featIdx]:
                self.clusterCenters[numClustSelected] = self.features[featIdx]
                numClustSelected += 1

    def vote(self, votedclass, votedcluster):
        #print '%i_%i'%(votedclass,votedcluster)
        for clssIdx in range(self.numclass):
            for clstrIdx in range(self.k):
                # mustLinkDistances
                if clssIdx == votedclass and clstrIdx != votedcluster:
                    self.classvotes[clssIdx][clstrIdx] += 1
                # cannotLinkDistances
                if clssIdx != votedclass and clstrIdx == votedcluster:
                    self.classvotes[clssIdx][clstrIdx] += 1


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
            int *cluster, float *feature2clusterDist, int numClusters, int numDim, int numPoints) {
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
				feature2clusterDist[valindex*numClusters + k] = distance;
              }
              cluster[valindex]=myCentroid;
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
            mod = compiler.SourceModule(kernel_code, options=["-ccbin",
                                                              "C:/Program Files (x86)/Microsoft Visual Studio 12.0/VC/bin/amd64"])
        else:
            mod = compiler.SourceModule(kernel_code)

        dataT = obs.T.astype(np.float32).copy()
        clusters = clusters.astype(np.float32)

        classIds = self.classId.astype(np.int32).copy()
        isFulls = self.isFull.astype(np.int32).copy()

        featureVote = gpuarray.zeros(points, dtype=np.int32)
        feature2clusterMinDist = gpuarray.zeros(len(self.features)*self.k, dtype=np.float32)

        kmeans_kernel = mod.get_function('cu_vq')
        kmeans_kernel(driver.In(dataT),
                      driver.In(clusters),
                      driver.In(classIds),
                      driver.In(isFulls),
                      featureVote,
                      feature2clusterMinDist,
                      np.int32(nclusters),
                      np.int32(dimensions),
                      np.int32(points),
                      grid=(blocks, 1),
                      block=(block_size, 1, 1),
                      )

        return featureVote.get(), feature2clusterMinDist.get()

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
            mod = compiler.SourceModule(kernel_code, options=["-ccbin",
                                                              "C:/Program Files (x86)/Microsoft Visual Studio 12.0/VC/bin/amd64"])
        else:
            mod = compiler.SourceModule(kernel_code)

        dataT = obs.T.astype(np.float32).copy()
        clusters = clusters.astype(np.float32)
        numIter = limits

        classIds = self.classId.astype(np.int32).copy()
        isFulls = self.isFull.astype(np.int32).copy()

        if (obs_code is None):
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
        points = obs.shape[0]
        dimensions = obs.shape[1]
        block_size = 512
        blocks = int(math.ceil(float(points) / block_size))
        kernel_code = kernel_code_template % {
            'DIMENSIONS': dimensions}

        import platform
        if '1003' in platform.node() and 'Linux' not in platform.system():
            mod = compiler.SourceModule(kernel_code, options=["-ccbin",
                                                              "C:/Program Files (x86)/Microsoft Visual Studio 12.0/VC/bin/amd64"])
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

    def cukmeans(self):
        self.clusterCenters = np.array(self.clusterCenters, copy=True)
        avg_dist = []
        iter = 0
        while iter < self.maxiter:
            # print "iteration number : ", iterNum
            print 'Iteration number %i (max %i)' % (iter, self.maxiter)
            self.classvotes = [[0] * self.k for i in
                               range(self.numclass)]  # votes of the classes for cluster, row for class
            self.clusterFeatures = [[] for i in range(self.k)]

            # if max 150 clusters, do the fast method
            if self.k <= 150:
                # compute membership and distances between features and code_book
                print "Apply K Means!"
                featureVote, self.featureClusterDist = self.cu_vq(self.features, self.clusterCenters)
            else:
                # Compute membership with little tasks
                print "Apply K Means!"
                limits = 0
                featureVote, self.featureClusterDist = self.cu_v2q(self.features, self.clusterCenters, None, None, limits)
                print limits, "-", limits + 50, "finished"
                limits += 50
                while limits < self.k:
                    featureVote, self.featureClusterDist = self.cu_v2q(self.features, self.clusterCenters, featureVote, distort, limits)
                    print limits, "-", limits + 50, "finished"
                    limits += 50

            print "Get Voting!"
            # each feature cast their negative votes
            for fidx in range(len(self.features)):
                cc = featureVote[fidx]
                if self.isFull[fidx]:
                    # if full vote
                    self.vote(self.classId[fidx], cc)
                else:
                    # else directly get assigned
                    if self.featureCluster[fidx] != cc:
                        noLabelChange = False
                    self.clusterFeatures[cc].append(fidx)

            # multiply votes by weight
            for voteidx, _ in enumerate(self.classvotes):
                self.classvotes[voteidx] = [self.currweight*v for v in self.classvotes[voteidx]]

            # Honor the voting
            print 'Assigning full features to clusters'
            for fidx in self.fullIndex:
                # iterate over each cluster and find the smallest distances
                bestdist, bestclstr = self.featureClusterDist[fidx*self.k + 0] + self.classvotes[self.classId[fidx]][0], 0
                for clstridx in range(self.k):
                    dist = self.featureClusterDist[fidx*self.k + clstridx] + self.classvotes[self.classId[fidx]][clstridx]
                    if dist < bestdist:
                        bestdist = dist
                        bestclstr = clstridx

                if self.featureCluster[fidx] != bestclstr:
                    noLabelChange = False

                self.featureCluster[fidx] = bestclstr
                self.clusterFeatures[bestclstr].append(fidx)

            if noLabelChange:
                print 'Break for no label change'
                break

            print 'Moving clusters'
            self.clusterMove2(self.features, self.clusterFeatures, self.clusterCenters)
            self.currweight = min(self.currweight + self.stepweight, self.maxweight)
            iter += 1
        return self.clusterFeatures, self.clusterCenters

        '''
        # Find the most voted cluster for every class
        classClusters = []
        i = 0

        for myClass in voteList:
            highestIdx = myClass.argmax()
            failsafe = self.k
            while (highestIdx in classClusters and failsafe > 0):
                myClass[highestIdx] = -1
                highestIdx = myClass.argmax()
                failsafe -= 1
            classClusters.append(highestIdx)
            i += 1

            if failsafe == 0:
                print "Failsafe actvated"
        '''

        '''
        # assign every full sketch to that cluster
        print "Reassign full sketches!"
        #featureVote = self.cu_av(self.features, self.clusterCenters, featureVote, self.class2cluster)
        # print nclasses, nclusters, len(voteList), len(instanceVotes)
        # print len(voteList[0]), len(voteList[1]), sum(voteList[0])

        # obs_code is the membership of points
        avg_dist.append(np.mean(distort, axis=-1))
        # recalc code_book as centroids of associated features
        has_members = []
        for i in np.arange(self.k):
            cell_members = np.compress(np.equal(featureVote, i), self.features, 0)
            if cell_members.shape[0] > 0:
                self.clusterCenters[i] = np.mean(cell_members, 0)
                has_members.append(i)
        # remove code_books that didn't have any members
        self.clusterCenters = np.take(self.clusterCenters, has_members, 0)
        self.currweight += self.stepweight
        iter += 1
        '''


    def clusterMove(self, features, clusterFeatures, clusterCenters):
        distsum = 0
        for cIdx in range(len(clusterFeatures)):
            featuresum = [0] * len(features[0])
            numFeaturesInCluster = len(clusterFeatures[cIdx])
            if numFeaturesInCluster != 0:
                for f in clusterFeatures[cIdx]:
                    featuresum = [self.features[f][idx] + featuresum[idx] for idx in range(len(self.features[0]))]

                featuresum = [featuresum[idx]/numFeaturesInCluster for idx in range(len(featuresum))]
                # assign the new cluster center
                clusterCenters[cIdx] = featuresum
            # what to do with empty cluster

    def clusterMove2(self, features, clusterFeatures, clusterCenters):
        has_members = []
        for i in np.arange(len(clusterFeatures)):
            #cell_members = np.compress(np.equal(self.featureCluster, i), features, 0)
            cell_members = [features[idx] for idx in self.clusterFeatures[i]]
            if len(cell_members) > 0:
                self.clusterCenters[i] = np.mean(cell_members, 0)
                has_members.append(i)
        # remove code_books that didn't have any members
        #self.clusterCenters[i] = np.take(self.clusterCenters[i], has_members, 0)