#ifndef KERNELS
#define KERNELS

#include </content/src/utils.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>

__global__ void get_bc(const int * edges_from, const int * edges_to, const int * edges, const int m, const int n, float * bc, int * Q1, int * Q2, int * Q1_len, int * Q2_len, int * dist, int * num_shortest_paths, int * S, int * S_ends, int * S_len, float * bc_local)
{
 int i = blockIdx.x*blockDim.x + threadIdx.x;
 // Initialize all global arrays and stuff to 0 

 for(int k = 0; k < (n + gridDim.x*blockDim.x - 1)/(gridDim.x*blockDim.x); k++)
 {
  if(k*gridDim.x*blockDim.x + i < n)
  {
   if(k==0 && i==0)
   {
    Q1_len[0] = 0;
    Q2_len[0] = 0;  // Empty queues
   }
   bc[i + k*gridDim.x*blockDim.x] = 0;
   bc_local[i + k*gridDim.x*blockDim.x] = 0;
   Q1[i + k*gridDim.x*blockDim.x] = 0;
   Q2[i + k*gridDim.x*blockDim.x] = 0;
   dist[i + k*gridDim.x*blockDim.x] = 0;
   num_shortest_paths[i + k*gridDim.x*blockDim.x] = 0;
   S[i + k*gridDim.x*blockDim.x] = 0;
   S_ends[i + k*gridDim.x*blockDim.x] = 0;
  }
 }

 __syncthreads();
 

 for(int source = 0; source < n; source++)
 {
  __syncthreads();
  // New node, so set dist, num_shortest paths and bc_local to be zero!
  
  for(int k = 0; k < (n + gridDim.x*blockDim.x - 1)/(gridDim.x*blockDim.x); k++)
  {
    if(k*gridDim.x*blockDim.x + i < n)
    {
      if(k==0 && i==0)
      {
      S_len[0] = 0;
      Q1_len[0] = 0;
      Q2_len[0] = 0;
      }
      bc_local[i + k*gridDim.x*blockDim.x] = 0;
      Q1[i + k*gridDim.x*blockDim.x] = 0;
      Q2[i + k*gridDim.x*blockDim.x] = 0;
      dist[i + k*gridDim.x*blockDim.x] = INT_MAX;
      num_shortest_paths[i + k*gridDim.x*blockDim.x] = 0;
      S[i + k*gridDim.x*blockDim.x] = 0;
      S_ends[i + k*gridDim.x*blockDim.x] = 0;
      }
  }

  __syncthreads();

  int max_penultimate_depth = 0;

  // Run the shortest path calculation, from the source node outwards
  
  if(i==0)
  {
   // Only setting the source node here
   Q1[0] = source;
   Q1_len[0] = 1;
   dist[source] = 0;
   num_shortest_paths[source] = 1;
   S[0] = source;
   S_ends[0] = 0;
   S_len[0] = 1;
  }
  // Now, Q1 has the source node, Q2 is empty, S, S_ends have been assigned

  int curr_depth = 0; // Depth is defined as the distance of the frontier from the source

  __syncthreads();

  while(1)
  {
   // There are still unexplored nodes. Everything till current depth is done
   // Find next frontier nodes in parallel

   for(int k = 0; k < (Q1_len[0] + gridDim.x*blockDim.x - 1)/(gridDim.x*blockDim.x); k++)
   {
    if(k*gridDim.x*blockDim.x + i < Q1_len[0])
    {
     for(int neighbors_idx = edges_from[Q1[k*gridDim.x*blockDim.x + i]]; neighbors_idx <= edges_to[Q1[k*gridDim.x*blockDim.x + i]]; neighbors_idx++)
     {
      // neighbors_idx is a neighbor of a node in the current frontier
      if(atomicCAS(&dist[edges[neighbors_idx]], INT_MAX, curr_depth + 1) == INT_MAX)
      {
       // It is an unexplored node
       int temp = atomicAdd(&Q2_len[0], 1);
       Q2[temp] = edges[neighbors_idx];
      }
      if(dist[edges[neighbors_idx]] == curr_depth + 1)
      {
       atomicAdd(&num_shortest_paths[edges[neighbors_idx]], num_shortest_paths[Q1[k*gridDim.x*blockDim.x + i]]);
      }
     }
    }
   }
   
   __syncthreads();

   int next_frontier_len = Q2_len[0];

   if(next_frontier_len == 0)
   {
    max_penultimate_depth = curr_depth - 1;
    break;
   }
   else
   {
    for(int k = 0; k < (next_frontier_len + gridDim.x*blockDim.x - 1)/(gridDim.x*blockDim.x); k++)
    {
     if(k*gridDim.x*blockDim.x + i < next_frontier_len)
     {
      // Move Q2 into Q1
      Q1[k*gridDim.x*blockDim.x + i] = Q2[k*gridDim.x*blockDim.x + i];
      S[k*gridDim.x*blockDim.x + i + S_len[0]] = Q2[k*gridDim.x*blockDim.x + i];
     }
    }
    __syncthreads();

    curr_depth = curr_depth + 1;
    S_ends[curr_depth] = S_ends[curr_depth - 1] + next_frontier_len;
    Q1_len[0] = next_frontier_len;
    S_len[0] = S_len[0] + next_frontier_len;
    Q2_len[0] = 0;
    __syncthreads();
   }
   __syncthreads();

  }
  
  while(max_penultimate_depth > 0)
  {
   for(int k = 0; k < (S_ends[max_penultimate_depth] - S_ends[max_penultimate_depth - 1] + gridDim.x*blockDim.x - 1)/(gridDim.x*blockDim.x); k++)
   {
    if(k*gridDim.x*blockDim.x + i < S_ends[max_penultimate_depth] - S_ends[max_penultimate_depth - 1])
    {
     int curr_frontier_node = S[S_ends[max_penultimate_depth - 1] + 1 + k*gridDim.x*blockDim.x + i];
     float curr_frontier_node_dependency = 0;
     int curr_frontier_node_paths = num_shortest_paths[curr_frontier_node];
     
     for(int neighbors_idx = edges_from[curr_frontier_node]; neighbors_idx <= edges_to[curr_frontier_node]; neighbors_idx++)
     {
      if(dist[edges[neighbors_idx]] == max_penultimate_depth + 1)
      {
       curr_frontier_node_dependency = curr_frontier_node_dependency + ((float)curr_frontier_node_paths)/(num_shortest_paths[edges[neighbors_idx]]) * (1.0 + bc_local[edges[neighbors_idx]]);
      }
     }
     bc_local[curr_frontier_node] = curr_frontier_node_dependency;
    }
   }

   __syncthreads();
   max_penultimate_depth = max_penultimate_depth - 1;
  }

  __syncthreads();

  // Dependency d_s(v) has been stored in bc_local, and so this will be parallelly accumulated in bc

  for(int k = 0; k < (n + gridDim.x*blockDim.x - 1)/(gridDim.x*blockDim.x); k++)
  {
   if(k*gridDim.x*blockDim.x + i < n)
   {
    bc[k*gridDim.x*blockDim.x + i] = bc[k*gridDim.x*blockDim.x + i] + bc_local[k*gridDim.x*blockDim.x + i];
   }
  }
  __syncthreads();
 }

}

float * work_efficient_bc_gpu(utils::graph g)
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
 int * d_edges_from = NULL;
 int * d_edges_to = NULL;
 float * d_bc = NULL;
 float * d_bc_local = NULL;
 int * Q1 = NULL;
 int * Q2 = NULL;
 int * Q1_len = NULL;
 int * Q2_len = NULL;
 int * S_len = NULL;
 int * dist = NULL;
 int * num_shortest_paths = NULL;
 int * S = NULL;
 int * S_ends = NULL;


 size_t size = g.n*sizeof(int);
 err = cudaMalloc((void**)&d_edges_from, size);
 if(err!=cudaSuccess){std::cout<<"1"<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMemcpy(d_edges_from, g.edges_from, size, cudaMemcpyHostToDevice);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMalloc((void**)&d_edges_to, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMemcpy(d_edges_to, g.edges_to, size, cudaMemcpyHostToDevice);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMalloc((void**)&Q1, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMalloc((void**)&Q2, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMalloc((void**)&dist, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMalloc((void**)&num_shortest_paths, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMalloc((void**)&S, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMalloc((void**)&S_ends, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 
 size = sizeof(int);
 err = cudaMalloc((void**)&Q1_len, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMalloc((void**)&Q2_len, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMalloc((void**)&S_len, size);
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
 
 cudaEvent_t start, stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);

 cudaEventRecord(start);
 get_bc<<<num_SMs, max_threads_per_block>>>(d_edges_from, d_edges_to, d_edges, g.m, g.n, d_bc, Q1, Q2, Q1_len, Q2_len, dist, num_shortest_paths, S, S_ends, S_len, d_bc_local);
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