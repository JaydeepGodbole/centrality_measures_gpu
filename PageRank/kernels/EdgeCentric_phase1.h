#include <cuda.h>
#include <cuda_runtime.h>


__global__ void  EdgeCentric_phase1(float* newpr,const float* oldpr,const int* sinkArray, int N,int countSink,float df){
    

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    while (idx < N){
        
        //Add page rank contributed by all sink nodes
        float sum = 0;
        for(int i=0; i<countSink; i++){
            int wId = sinkArray[i];
            sum+= oldpr[wId]/N;
        }

        newpr[idx] = (df*sum) + (1-df)/N;

        idx += gridDim.x*blockDim.x;
    }
}
