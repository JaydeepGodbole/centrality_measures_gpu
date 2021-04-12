#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 256

__global__ void  EdgeCentric_phase1_optmized(float* newpr,const float* oldpr,const int* sinkArray, int N,int countSink,float df){
    

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float d_sink[TILE_DIM];

    while (idx < N){
        
        //Add page rank contributed by all sink nodes using shared memory
        float sum = 0;

        int tx= threadIdx.x;
        int M = (countSink/TILE_DIM)+((countSink%TILE_DIM)!=0);
        
        for(int m=0;m<M;m++){
        
            if(m*TILE_DIM+tx<countSink){
                int wId = sinkArray[m*TILE_DIM+tx];;
                d_sink[tx] = oldpr[wId]/N;
            }
            else
                d_sink[tx]=0;
                
            __syncthreads () ;
            for(int k=0;k<TILE_DIM;k++)
                sum += d_sink[k];
            
            __syncthreads () ;
        }
        
        newpr[idx] = (df*sum) + (1-df)/N;

        idx += gridDim.x*blockDim.x;
    }
}