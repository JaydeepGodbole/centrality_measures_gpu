#ifndef KERNELS
#define KERNELS

#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../utils.h"

using namespace std;

void printGraph(int *adjr, int *adjc, int V)
{
    for (int v = 1; v <= V; ++v)
    {
        cout << "\n Adjacency list of vertex "
             << v-1 << "\n head ";
        for (int i = adjr[v-1]; i<adjr[v];i++)
        {   
            
            cout << "-> " << adjc[i];
            
        }
        printf("\n");
    }
}

float device_time_taken;

void printTime(float ms) {
    int h = ms / (1000*3600);
    int m = (((int)ms) / (1000*60)) % 60;
    int s = (((int)ms) / 1000) % 60;
    int intMS = ms;
    intMS %= 1000;

    printf("Time Taken (Parallel) = %dh %dm %ds %dms\n", h, m, s, intMS);
    printf("Time Taken in milliseconds : %d\n", (int)ms);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
 __device__ double atomicAdd(double* address, double val) 
{ 
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    } while (assumed != old);

    return __longlong_as_double(old);

}
#endif

__global__ void kernel_node(int garbage, const int *R, const int *C, double *bc,const int nodes,int *d, int *sigma, double *delta, int *reverse_stack)
{
    __shared__ int position;
    
    
    // Used to store the source vertex            
    __shared__ int s;

    //__shared__ int end_pos;
    
    int idx = threadIdx.x; // blockDim.x * blockIdx.x +
    if (idx == 0) {
        // Initializing source
        s = 0;
        //end_pos = 1;
        //reverse_bfs_limit[0] = 0;
    }
    if(idx<nodes)
    {

        __syncthreads();
        
          
        while (s < nodes) {
            __syncthreads();
            
            // ============== Vertex parallel method for BFS ========================
                    
            //Initialize d and sigma
            for(int v=idx; v<nodes; v+=blockDim.x) {
                
                if(v == s) {
                    d[v] = 0;
                    sigma[v] = 1;
                }
                else {
                    d[v] = INT_MAX;
                    sigma[v] = 0;
                }
                delta[v] = 0;
            }
            __syncthreads();
            __shared__ int current_depth;
            __shared__ bool done;

            // ============== INIT ========================
                    
            if(idx == 0) {
                done = false;
                current_depth = 0;
                position = 0;
            }
            __syncthreads();
            
            // SP Calc 
            while(!done)
            {
                __syncthreads();
                done = true;
                __syncthreads();
                
                for(int v=idx; v<nodes; v+=blockDim.x) {
                    if(d[v] == current_depth) {
                        
                        // ============== Storing nodes for reverse BFS ========================
                    
                        int t = atomicAdd(&position,1);
                        reverse_stack[t] = v;

                        // ============== Relaxation step to find minimum distance ========================
                    
                        for(int r=R[v]; r<R[v+1]; r++) {
                            int w = C[r];
                            if(d[w] == INT_MAX) {
                                d[w] = d[v] + 1;
                                done = false;
                            }
                            if(d[w] == (d[v] + 1)) {
                                atomicAdd(&sigma[w],sigma[v]);
                            }
                        }
                    }
                }
                __syncthreads();
                if(idx == 0){
                    current_depth++;
                    //reverse_bfs_limit[end_pos] = position;
                    //++end_pos;
                }
            }

            // printf("%f\n", bc[1]);
       
            __syncthreads();
            

            if(idx == 0)
            {
                
                // printf("%f\n", bc[1]);   
                for(int itr1 = nodes - 1; itr1 >= 0; --itr1){
                    for(int itr2 = R[reverse_stack[itr1]]; itr2 < R[reverse_stack[itr1] + 1]; ++itr2){
                        int consider = C[itr2];
                        if(d[consider] == d[reverse_stack[itr1]]-1){
                            delta[consider] += ( ((float)sigma[consider]/sigma[reverse_stack[itr1]]) * ((float)1 + delta[reverse_stack[itr1]]) ); 
                        }
                    }
                    if(reverse_stack[itr1] != s)
                    {
                        // bc[reverse_stack[itr1]] += delta[reverse_stack[itr1]]/(nodes*(nodes-1));
                        atomicAdd(&bc[reverse_stack[itr1]],delta[reverse_stack[itr1]]/(nodes*(nodes-1)));
                        // printf("%f\n", bc[reverse_stack[itr1]]);
                    }
                }
            }

            // ============== Incrementing source ========================
                    
            __syncthreads();
            if (idx == 0) {
                s += 1;
            }
           
        }
    }
} 

double *betweenness_node(Graph g)
{
    int V,E;

    /* calculate betweenness centrality score using bfs */
    
    int *adjr,*adjc;
    adjr = g.getAdjacencyListPointers();
    adjc = g.getAdjacencyList();

    V = g.getNodeCount();
    E = g.getEdgeCount();
    //printGraph(adjr,adjc,V);

    double *score;
    score = (double*)malloc(V*sizeof(double));
    for (int i = 0; i < V; ++i)
    {
        score[i] = 0;
    }

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);

    int *adj_r,*adj_c;
    int *d_d, *d_sigma, *d_reverse_stack;
    double *d_delta;

    // Allocating memory via cudamalloc

    cudaMalloc((void**)&d_d, sizeof(int) * V);
    cudaMalloc((void**)&d_sigma, sizeof(int) * V);
    cudaMalloc((void**)&d_reverse_stack, sizeof(int) * V);
    cudaMalloc((void**)&d_delta, sizeof(double) * V);


    double *result = (double*)malloc(V*sizeof(double));

    err = cudaMalloc((void **)&adj_r,(V+1)*sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&adj_c,(2*E+1)*sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    double *score_A;
    err = cudaMalloc((void **)&score_A,V*sizeof(double));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device score A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(adj_r, adjr, (V+1)*sizeof(int), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy adjacent matrix from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(adj_c, adjc, (2*E+1)*sizeof(int), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy adjacent matrix from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(score_A, score, V*sizeof(double), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy score from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    int threadsPerBlock = 512;
    int blocksPerGrid = ((V-1)/threadsPerBlock)+1;//(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    cudaEvent_t device_start, device_end;
    cudaEventCreate(&device_start);
    cudaEventCreate(&device_end);
    cudaEventRecord(device_start);


    kernel_node<<<blocksPerGrid, threadsPerBlock>>>(0,adj_r, adj_c, score_A, V,  d_d, d_sigma, d_delta, d_reverse_stack);
    
    cudaDeviceSynchronize();

    cudaEventRecord(device_end);
    cudaEventSynchronize(device_end);
    cudaEventElapsedTime(&device_time_taken, device_start, device_end);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(result, score_A, V*sizeof(double), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(adj_r);
    err = cudaFree(adj_c);
    err = cudaFree(score_A);
    
    cudaFree(d_sigma);
    cudaFree(d_d);
    cudaFree(d_delta);
    cudaFree(d_reverse_stack);

    printTime(device_time_taken);

    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(adjr);
    free(adjc);
    free(score);

    return result;
}
#endif