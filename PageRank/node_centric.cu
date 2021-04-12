#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include<vector>
#include<utility>
#include<algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>

using namespace std;

float df = 0.85f;

struct Graph{
    int N; // number of nodes
    int countSink; //number of sink nodes
    int E;
    int *ptrarray;
    int *sinkArray; //sink nodes
    int *adjListArray;  // adjacency list of transposed graph
    int *outdegree; // contains outdegree of all nodes
    int *indegree; //contains indegree of all nodes
    float *pr; // page rank values
};


Graph* buildGraph(vector<pair<int,int>>& edges, int E,int V)
{   
    Graph* G = new Graph();
    G->N = V;
    G->E = E;
    G->ptrarray = new int[V+1];
    G->adjListArray = new int[E]; 
    G-> outdegree = new int[V]();
    G-> indegree = new int[V]();
    G->pr = new float[V];

    
    sort(edges.begin(),edges.end(),[](pair<int,int>& e1,pair<int,int>& e2){
        return (e1.second==e2.second)? (e1.first<e2.first): (e1.second<e2.second);
    });
 
    for(int i=0;i<E;i++){
        G->adjListArray[i] = edges[i].first;
        G->outdegree[edges[i].first]++;
        G->indegree[edges[i].second]++;
    }
 
    int x=0;
    int count = 0;
    for(int i=0;i<=V;i++){
        G->ptrarray[i] = x;
        if (i<V){
            x+= G->indegree[i];
            if (G->outdegree[i] == 0){
                count += 1;
            }
        }
    }

    G-> countSink = count;
    G-> sinkArray = new int[count]();
    x = 0;
    for( int i = 0; i < V; i++){
        if (G->outdegree[i] == 0){
            G-> sinkArray[x] = i;
            x+= 1;
        }
    }
    
    return G;
}


Graph* readgraph(const char* file){

    FILE *in_file = fopen(file, "r");

    int E,V = 0;
    fscanf(in_file, "%d %d", &E,&V);

    vector<pair<int,int>> edges(E);
    for (auto& e:edges) {
        fscanf(in_file, "%d %d", &e.first, &e.second);
    }
    fclose(in_file);
    
    return buildGraph(edges, E, V);
} 

// Stores the page rank values of the given Graph structure in output.txt
void storePageRank(Graph* graph,const char* file)
{
    FILE *out_file = fopen(file, "w");
    for (int i=0; i<graph->N; i++) {
        fprintf(out_file, "%f\n", graph->pr[i]);
    }
    fclose(out_file);
}

// Initialises the page rank values of the Graph structure to 1
void initialisePageRank(Graph *graph)
{

    for (int i=0; i<graph->N; i++) {
        graph->pr[i] = 1.0f;
    }
}


__global__ void  UpdatePagerank(const int* ptrArray,const int* outdegreeArray, const int* sinkArray,const int* adjListArray, 
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


void PageRank(Graph* G,int iter,float df,int blocksPerGrid,int threadsPerBlock){


    cudaError_t err = cudaSuccess;

    cout<<"Initialize arrays in device memory\n";
    
    //Allocate the device ptrArray
    int* d_ptrArray = NULL;
    err = cudaMalloc((int **)&d_ptrArray, (G->N+1)*sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector ptrArray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Allocate the device ptrArray
    int* d_outdegreeArray = NULL;
    err = cudaMalloc((int **)&d_outdegreeArray, (G->N)*sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector outdegreeArray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Allocate the device sinkArray
    int* d_sinkArray = NULL;
    err = cudaMalloc((int **)&d_sinkArray, G->countSink * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector ptrArray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Allocate the device adjListArray
    int* d_adjListArray = NULL;
    err = cudaMalloc((int **)&d_adjListArray, G->E*sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector adjListArray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Allocate the device oldpr
    float* d_oldpr = NULL;
    err = cudaMalloc((float **)&d_oldpr, G->N*sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector oldpr (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Allocate the device newpr output array
    float* d_newpr = NULL;
    err = cudaMalloc((float **)&d_newpr, G->N*sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector newpr (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    //copy input data from host to device
    cout<<"Copy input data from host memory to CUDA device memory\n";
    err = cudaMemcpy(d_ptrArray, G->ptrarray, (G->N+1)*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector ptrArray from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_outdegreeArray, G->outdegree, (G->N)*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector outdegree from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_sinkArray, G->sinkArray, G->countSink * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector sinkArray from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_adjListArray, G->adjListArray,  G->E*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector adjListArray from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_oldpr, G->pr, G->N*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector pr from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // grid and block dimension
    dim3 grid(blocksPerGrid,1,1);
    dim3 block(threadsPerBlock,1,1);
    
    while(iter--){
        // Launch the PageRank Update CUDA Kernel
        UpdatePagerank<<<grid, block>>>(d_ptrArray, d_outdegreeArray, d_sinkArray, d_adjListArray, d_oldpr, 
                                        d_newpr,G->N,df, G->countSink);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch UpdatePageRank kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        
        //update the oldpr array
        err = cudaMemcpy(d_oldpr,d_newpr,G->N*sizeof(float),cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy newpr array from device to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    err = cudaMemcpy(G->pr,d_oldpr,G->N*sizeof(float),cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy pr array from device to Host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    //free up allocated memory
    err = cudaFree(d_ptrArray);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector ptrArray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_adjListArray);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector adjListArray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_oldpr);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector oldpr (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_newpr);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector newpr (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_outdegreeArray);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector outdegreeArray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_sinkArray);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector sinkArray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Reset the device
    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}


int main(){
    char input_file[] = "input.txt";
    char output_file[] = "nodeCentric_output.txt";
    Graph* G = readgraph(input_file);
    initialisePageRank(G);
    int threadsPerBlock = 256;
    int blocksPerGrid = (G->N+threadsPerBlock-1)/threadsPerBlock;
    PageRank(G,1000,df,blocksPerGrid,threadsPerBlock);
    cout<<"PageRank calculation done!!"<<endl;

    storePageRank(G,output_file);
    return 0;
}
