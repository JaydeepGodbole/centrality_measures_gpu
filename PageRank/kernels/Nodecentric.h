#include <cuda.h>
#include <cuda_runtime.h>

__global__ void  Nodecentric(const int* ptrArray,const int* outdegreeArray, const int* sinkArray,const int* adjListArray, 
                                const float* oldpr, float* newpr,int N, float df, int countSink){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    while(idx<N){
        float sum = 0;
        for(int w=ptrArray[idx];w<ptrArray[idx+1];w++){
            int wId = adjListArray[w];
            if (w == ptrArray[idx+1])   //indegree = 0
            break;
            int wOutDegree = outdegreeArray[wId];
            sum += oldpr[wId]/wOutDegree;
        }

        //Add page rank contributed by all sink nodes
        for(int i=0; i<countSink; i++){
            int wId = sinkArray[i];
            sum+= oldpr[wId]/N;
        }
        newpr[idx] = (df*sum) + (1-df)/N;
        idx += gridDim.x*blockDim.x;
    }
}
