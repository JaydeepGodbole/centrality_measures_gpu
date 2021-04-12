#include <cuda.h>
#include <cuda_runtime.h>


__global__ void EdgeCentric_phase2(const int* outdegreeArray, const int* edgeArray1, const int* edgeArray2, 
                                    const float* oldpr, float* newpr, int E, float df){
    
    
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    while (idx < E){
        int w = edgeArray1[idx];
        int v = edgeArray2[idx];

        float* arg1 = &newpr[v];
        float arg2 = df*oldpr[w]/outdegreeArray[w];
        float arg5 = atomicAdd(arg1, arg2);

        idx += gridDim.x*blockDim.x;
    }  
}
