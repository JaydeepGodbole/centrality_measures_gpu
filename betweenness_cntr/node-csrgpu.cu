#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "graph.h"

using namespace std;

// #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// #else
// __device__ double atomicAdd(double* address, double val) 
// { 
//     unsigned long long int* address_as_ull =
//                               (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;

//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed,
//                         __double_as_longlong(val +
//                                __longlong_as_double(assumed)));

//     // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//     } while (assumed != old);

//     return __longlong_as_double(old);

// }
// #endif

void addEdge(int *adj, int u, int v, int t)
{
    adj[u*t+v] = 1;
    adj[v*t+u] = 1;
}

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

// __device__ struct ls
// {   
//     int value;
//     ls *next;
// };

// __device__ ls *pushq(ls *q,int val)
// {   
//     ls *head = q;
//     ls *temp = (ls*)malloc(sizeof(ls));
    
//     if(q==NULL)
//     {       

//         q = (ls*)malloc(sizeof(ls));
//         q->value = val;
//         q->next = NULL;
//         return q;
//     }
//     else
//     {
//         while(q->next!=NULL)
//         {
            
//             q = q->next;
//         }
        
//         temp->value = val;
//         temp->next = NULL;
//         q->next = temp;

//     }

//     return head;
// }

// __device__ int qfront(ls *q)
// {   
//     if(q!=NULL)
//         return q->value;
//     return -1;
// }

// __device__ int qfrontt(ls *q)
// {      
//     if(q==NULL)
//         return -1;
//     while(q->next!=NULL)
//     {
//         q = q->next;
//     }
//     return q->value;
// }
// __device__ ls *qpop(ls *q)
// {
//     if(q!=NULL)
//     {
//         ls *temp;// = (ls*)malloc(sizeof(ls));
//         temp = q;
//         q = q->next;
//         free(temp);
//         return q;
//     }
//     return q;
// }

// __device__ ls *qpopt(ls *q)
// {
//     if(q!=NULL)
//     {
//         ls *temp,*head;// = (ls*)malloc(sizeof(ls));
//         temp = q;
//         head = q;
//         ls *fr;
//         fr = NULL;
//         if(q->next==NULL)
//             return NULL;
//         while(q->next!=NULL)
//         {   
//             fr = q;
//             q = q->next;
//             temp = q;
//         }
//         fr->next = NULL;
//         free(temp);
//         return head;
//     }
//     return q;
// }

// __device__ void bfs(int src,const int *adjr, const int *adjc, const int v,
//          ls **pred, int *dist, ls **visitstack, int *sigma, float *val)
// {
//     ls *queue = NULL;
    

//     bool *visited;
//     visited = (bool*)malloc(v*sizeof(bool));
//     if (visited == NULL) 
//     {   

//         return;
//     }


//     memset(visited, false, v * sizeof(bool));
//     memset(dist, 1000, v * sizeof(int));
//     //memset(pred, -1, v * sizeof(int));

//     visited[src] = true;
//     dist[src] = 0;
//     sigma[src] = 1;

//     queue = pushq(queue,src);
    
//     // __syncthreads();

//     // standard BFS algorithm
//     *val = 0;
//     float closeness = 0;

//     while (queue!=NULL) 
//     {
//         int u = qfront(queue);
//         queue = qpop(queue);
//         *visitstack = pushq(*visitstack,u);
//         //printf("Thread %d : queue %d \n",src,qfront(visitstack));
//         //break;

//         closeness += dist[u];
//         for (int j = adjr[u]; j < adjr[u+1]; j++) 
//         {
//             if (!visited[adjc[j]]) 
//             {   
//                 visited[adjc[j]] = true;
//                 dist[adjc[j]] = dist[u] + 1;
//                 //pred[j] = u;
//                 queue = pushq(queue,adjc[j]);

//                 if (dist[adjc[j]] == dist[u]+1)
//                 {
//                     pred[adjc[j]] = pushq(pred[adjc[j]],u);
//                     atomicAdd(&sigma[adjc[j]],sigma[u]);

//                 }
//             }

            
//         }
        
//     }
//     // printf("Error here\n");
//     __syncthreads();

//     if(closeness!=0)
//     {
//         *val = 1.0/closeness;
//     }

//     free(visited);
//     free(queue);
//     return;
// }

__global__ void sd(int garbage, const int *R, const int *C, float *bc,const int nodes,int *d, int *sigma, float *delta, int *reverse_stack)
{
 //    int th = blockDim.x * blockIdx.x + threadIdx.x ;

 //    if(th>=v)
 //    {
 //        return;
 //    }

 //    if(th<v)
 //    {       
        
 //        // int *dist, *sigma;
 //        ls **pred;
 //        // float *delta;

 //        pred = (ls**)malloc(v*sizeof(ls*));
 //        // dist = (int*)malloc(v*sizeof(int));
 //        // sigma = (int*)malloc(v*sizeof(int));
 //        // delta = (float*)malloc(v*sizeof(float));

 //        if (sigma == NULL) 
 //        {   
 //            // printf("hi 1");
 //            return;
 //        }
 //        if (pred == NULL) 
 //        {   
 //            // printf("hi ");
 //            return;
 //        }
 //        if (dist == NULL) 
 //        {   
 //            // printf("hi ");
 //            return;
 //        }

 //        // printf("Thread : %d \n",th);
 //        memset(sigma+th*v, 0, v * sizeof(int));
 //        memset(delta+th*v, 0.0, v * sizeof(float));

 //        ls *visitstack=NULL;

 //        float *val;
 //        float x = 0;
 //        val = &x;
        
 //        // printf("Thread : %d \n",th);
 //        bfs(th, adjr, adjc, v, pred, dist+th*v, &visitstack, sigma+th*v, val);
 //        // printf("Thread : %d \n",th);

 //        printf("Thread %d : queue %d \n",th,qfront(visitstack));

 //        __syncthreads();
 //        while (visitstack!=NULL) 
 //        {   

 //            int w = qfrontt(visitstack);
 //            visitstack = qpopt(visitstack);
 //            //printf("%d ",w);
            
 //            // For each predecessors of node w, do the math!
 //            float c;
 //            while(pred[w]!=NULL) 
 //            {   
 //                int p = qfront(pred[w]);
 //                pred[w] = qpop(pred[w]);
 //                c = ((float) sigma[th*v+p] / (float) sigma[th*v+w]) * (1.0 + delta[th*v+w]);
 //                // printf("%f ",c);
 //                delta[th*v+p] += c;

 //            }
        
 //            // Node betweenness aggregation part.
 //            if (w != th) 
 //            {
 //                //score[w] += delta[w];
 //                atomicAdd(&score[w],delta[th*v+w]);
 //            }
 //        }
 //        free(visitstack);
 //        free(pred);
 //        // free(dist);
 //        // free(delta);
 //        // free(sigma);
 //        __syncthreads();        
        
	// }
 //    // printf("Hi here ");
    
 //    return;


    __shared__ int position;
    
    
    // Used to store the source vertex            
    __shared__ int s;

    //__shared__ int end_pos;
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
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
            // Parallel Vertex Parallel implementation (uncomment the following lines and comment the ones below)
       
            __syncthreads();
            // atomicSub(&end_pos,2);
            // for(int itr1 = end_pos; itr1 >= 0; --itr1){
            //     for(int itr2 = reverse_bfs_limit[itr1] + idx; itr2 < reverse_bfs_limit[itr1+1]; itr2+=blockDim.x){
            //         // reverse_stack[itr2] is one node
            //         for(int itr3 = R[reverse_stack[itr2]]; itr3 < R[reverse_stack[itr2] + 1]; ++itr3){
            //             int consider = C[itr3];
            //             // C[itr3] other node
            //             if(d[consider] == d[reverse_stack[itr2]]-1){
            //                 delta[consider] += ( ((float)sigma[consider]/sigma[reverse_stack[itr2]]) * ((float)1 + delta[reverse_stack[itr2]]) ); 
            //             }
            //         }
            //         if(reverse_stack[itr2] != s){
            //             bc[reverse_stack[itr2]] += delta[reverse_stack[itr2]];
            //         }

            //     }
            //     __syncthreads();
            // }

            
            // Serialized Vertex Parallel implementation. Comment the following for parallel implementation

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
                    if(reverse_stack[itr1] != s){
                        bc[reverse_stack[itr1]] += delta[reverse_stack[itr1]]/(nodes*(nodes-1));
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

void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
    return;
}

int main()
{
    int V,E;

    /* calculate betweenness centrality score using bfs */
 	
    Graph g;

    g.readGraphfile("as_733_csr.txt");  //facebook_combined
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

    // cudaDeviceProp devProp;
    // cudaGetDeviceProperties(&devProp, 0);
    // printDevProp(devProp);

    // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);

    int *adj_r,*adj_c;
    // int *sigma,*dist;
    // float *delta;

    int *d_d, *d_sigma, *d_reverse_stack;
    float *d_delta;

    // Allocating memory via cudamalloc

    cudaMalloc((void**)&d_d, sizeof(int) * V);
    cudaMalloc((void**)&d_sigma, sizeof(int) * V);
    cudaMalloc((void**)&d_reverse_stack, sizeof(int) * V);
    cudaMalloc((void**)&d_delta, sizeof(float) * V);


    float *result = (float*)malloc(V*sizeof(float));

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

    float *score_A;
    err = cudaMalloc((void **)&score_A,V*sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device score A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // err = cudaMalloc((void **)&sigma,(V*V)*sizeof(int));

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // err = cudaMalloc((void **)&dist,(V*V)*sizeof(int));

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // err = cudaMalloc((void **)&delta,V*V*sizeof(float));

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to allocate device score A (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }



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
    err = cudaMemcpy(score_A, score, V*sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy score from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    int threadsPerBlock = 1024;
    int blocksPerGrid = ((V-1)/threadsPerBlock)+1;//(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    cudaEvent_t device_start, device_end;
    cudaEventCreate(&device_start);
    cudaEventCreate(&device_end);
    cudaEventRecord(device_start);


    sd<<<blocksPerGrid, threadsPerBlock>>>(0,adj_r, adj_c, score_A, V,  d_d, d_sigma, d_delta, d_reverse_stack);
    
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
    err = cudaMemcpy(result, score_A, V*sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(adj_r);
    err = cudaFree(adj_c);
    err = cudaFree(score_A);
    // err = cudaFree(sigma);
    // err = cudaFree(dist);
    // err = cudaFree(delta);
    cudaFree(d_sigma);
    cudaFree(d_d);
    cudaFree(d_delta);
    cudaFree(d_reverse_stack);


    ofstream myfile;
    char *filename = "output.txt";
    myfile.open (filename);

    for(int j=0;j<V;j++)
    {  
        //if(result[j]>0)
        myfile<<"The Score : "<<(result[j])<<endl;
    }

    myfile.close();

    printTime(device_time_taken);

    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(result);
    free(score);
    free(adjr);
    free(adjc);


    return 0;
}


/*
-------------------------------
Example CSR format for
nodes = 5
edges = 5
-------------------------------
nodeCount edgeCount
row offset
column indices
-------------------------------
5 5
0 2 5 7 9 10 
2 3 3 2 4 0 1 1 0 1 
-------------------------------



Enter the nodes and edges count :
5
7
Enter the row adjacency list:
0 3 7 9 12 15
Enter the column adjacency list:
0 1 4 0 2 3 4 1 3 1 2 4 0 1 3


*/