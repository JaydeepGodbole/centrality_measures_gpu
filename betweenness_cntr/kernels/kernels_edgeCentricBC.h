%%cuda --name kernels.h
#ifndef KERNELS
#define KERNELS

#include </content/src/utils.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>

__global__ void getBC(const int * nhbrs, const int * froms, const int m, const int n, float * nodeBC, int * distance, int * numSPs, float * dependency, bool * predecessor)
{
  int nedge = m;
  int nnode = n;
  
  for (int nid = threadIdx.x; nid < nnode; nid += blockDim.x) 
    {
      nodeBC[nid] = 0.0;
    }
  __syncthreads();
 
 for (int source = 0; source < nnode; source++)
 {
  for (int eid = threadIdx.x; eid < 2 * nedge; eid += blockDim.x) 
  {
    int from = froms[eid];
    if(from == source)
    {
      numSPs[from] = 1;
      distance[from] = 0;  
    }
    else
    {
      numSPs[from] = 0;
      distance[from] = -1;  
    }        
    predecessor[eid] = false;
    dependency[from] = 0;
  }
  __syncthreads();
 
  __shared__ bool done;
 
  int d = 0;
  done = false;
  while (!done){
    __syncthreads();
    done = true;
    d++;
    __syncthreads();
    for(int eid = threadIdx.x; eid < 2 * nedge; eid += blockDim.x){
      int from = froms[eid];
      if(distance[from]==d){
        int nhbr = nhbrs[eid];
        int nhbrDist = distance[nhbr];
        if (nhbrDist == -1)
        {
          distance[nhbr] = d + 1;
          nhbrDist = d + 1;
          done = false;
        }
        if(nhbrDist < d)
        {
          predecessor[eid] = true;
        }
        if(nhbrDist == d + 1)
        {
          atomicAdd(&numSPs[nhbr], numSPs[from]);
        }
      }
     
    }
    __syncthreads();
  }
  __syncthreads();

  while (d > 1){
    for (int eid = threadIdx.x; eid < 2 * nedge; eid += blockDim.x) 
    {
      int from = froms[eid];
      if(distance[from] == d)
      {
        if (predecessor[eid])
        {
          int nhbr = nhbrs[eid];
          float delta = (1.0 + dependency[from]) * (numSPs[nhbr] / numSPs[from]);
          atomicAdd(&dependency[nhbr], delta);
        }
      }
    }
    d--;
    __syncthreads();
  }
  
  __syncthreads();
 
  for (int nid = threadIdx.x; nid < nnode; nid += blockDim.x){
    nodeBC[nid] = nodeBC[nid] + dependency[nid];
  }
  __syncthreads();
}
}

float * edge_centric_bc_gpu(utils::graph g)
{
 cudaDeviceProp prop;
 cudaError_t err = cudaSuccess;
 err = cudaGetDeviceProperties(&prop, 0);
 if(err != cudaSuccess)
 {
  std::cout << "Failed" << std::endl;
  exit(EXIT_FAILURE);
 }
 std::cout << "Chosen Device: " << prop.name << std::endl;
 std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
 std::cout << "Number of Streaming Multiprocessors: " << prop.multiProcessorCount << std::endl;
 std::cout << "Size of Global Memory: " << prop.totalGlobalMem/(float)(1024*1024*1024) << " GB"<< std::endl;

 int max_threads_per_block = prop.maxThreadsPerBlock;
 int num_SMs = prop.multiProcessorCount;
 
 int * d_edges = NULL;
 int * d_froms = NULL;
 float * d_bc = NULL;
 float * d_bc_local = NULL;
 int * dist = NULL;
 int * num_shortest_paths = NULL;
 bool * predecessor = NULL;
 

 size_t size = g.n*sizeof(int);
 err = cudaMalloc((void**)&dist, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMalloc((void**)&num_shortest_paths, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 
 size = g.n*sizeof(float);
 err = cudaMalloc((void**)&d_bc, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMalloc((void**)&d_bc_local, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 
 size = 2*g.m*sizeof(int);
 err = cudaMalloc((void**)&d_edges, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMemcpy(d_edges, g.edges, size, cudaMemcpyHostToDevice);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMalloc((void**)&d_froms, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMemcpy(d_froms, g.froms, size, cudaMemcpyHostToDevice);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}

 size = 2*g.m*sizeof(bool);
 err = cudaMalloc((void**)&predecessor, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 
 cudaEvent_t start, stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);

 cudaEventRecord(start);
 getBC<<<num_SMs, max_threads_per_block>>>(d_edges, d_froms, g.m, g.n, d_bc, dist, num_shortest_paths, d_bc_local, predecessor);
 cudaEventRecord(stop);
 cudaEventSynchronize(stop);
 float milliseconds = 0;
 cudaEventElapsedTime(&milliseconds, start, stop);
 
 std::cout << "The running time is " << milliseconds << "milliseconds" << std::endl;
 
 float * bc_calculated = new float[g.n];

 for(int i = 0; i < g.n; i++)
 {
  bc_calculated[i] = 1.0;
 }

 size = g.n*sizeof(float);
 err = cudaMemcpy(bc_calculated, d_bc, size, cudaMemcpyDeviceToHost);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}

 return bc_calculated;

}

#endif