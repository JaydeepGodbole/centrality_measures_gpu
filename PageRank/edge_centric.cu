#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include<vector>
#include<utility>
#include<algorithm>
#include <cstdio>
#include <fstream>
#include <math.h>
#include "kernels/EdgeCentric_phase1.h"
#include "kernels/EdgeCentric_phase2.h"

using namespace std;

#define BLOCK_DIM 256
float df = 0.85f;


struct Graph{
    int N; // number of nodes
    int countSink; //number of sink nodes
    int E;
    int *outdegree; // contains outdegree of all nodes
    int *indegree; //contains indegree of all nodes
    int *sinkArray;
    int *edgeArray1;
    int *edgeArray2;
    float *pr; // page rank values
};


Graph* buildGraph(vector<pair<int,int>>& edges, int E,int V)
{   
    Graph* G = new Graph();
    G->N = V;
    G->E = E;
    G-> outdegree = new int[V]();
    G->edgeArray1 = new int[E];
    G->edgeArray2 = new int[E];
    G->pr = new float[V];

    for(int i=0;i<E;i++){
        G->outdegree[edges[i].first]++;
        G->edgeArray1[i] = edges[i].first;
        G->edgeArray2[i] = edges[i].second;
    }
    

    G->countSink = 0;
    for(int i=0;i<V;i++){
        if (G->outdegree[i] == 0){
            G->countSink ++;
        }
        
    }

    G->sinkArray = new int[G->countSink]();
    int x = 0;
    for( int i = 0; i < V; i++){
        if (G->outdegree[i] == 0){
            G->sinkArray[x] = i;
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



void PageRank_edge_centric(Graph* G,int iter,float df,int blocksPerGrid,int threadsPerBlock){

    
    cudaError_t err = cudaSuccess;

    cout<<"Initialize arrays in device memory\n";
    
    int* d_outdegreeArray = NULL;
    err = cudaMalloc((int **)&d_outdegreeArray, (G->N)*sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector outdegreeArray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Allocate the device edgeArray1
    int* d_edgeArray1 = NULL;
    err = cudaMalloc((int **)&d_edgeArray1, G->E*sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector edgeArray1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    //Allocate the device edgeArray2
    int* d_edgeArray2 = NULL;
    err = cudaMalloc((int **)&d_edgeArray2, G->E*sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector edgeArray2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Allocate the device sinkArray
    int* d_sinkArray = NULL;
    err = cudaMalloc((int **)&d_sinkArray, G->countSink * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector sinkArray (error code %s)!\n", cudaGetErrorString(err));
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
    
    err = cudaMemcpy(d_outdegreeArray, G->outdegree, (G->N)*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector outdegree from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_edgeArray1, G->edgeArray1,  G->E*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector edgeArray1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_edgeArray2, G->edgeArray2,  G->E*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector edgeArray2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_sinkArray, G->sinkArray, G->countSink * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector sinkArray from host to device (error code %s)!\n", cudaGetErrorString(err));
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
        
        EdgeCentric_phase1<<<grid, block>>>(d_newpr,d_oldpr,d_sinkArray, G->N,G->countSink,df);
        EdgeCentric_phase2<<<grid, block>>>(d_outdegreeArray, d_edgeArray1, d_edgeArray2, 
                                    d_oldpr, d_newpr, G->E, df);
        
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
    err = cudaFree(d_edgeArray1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector edgeArray1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_edgeArray2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector edgeArray2 (error code %s)!\n", cudaGetErrorString(err));
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

    char input_file[] = "./Dataset/amazon.txt";
    char output_file[] = "edgeCentric_output.txt";
    Graph* G = readgraph(input_file); 
    initialisePageRank(G);
    int threadsPerBlock = BLOCK_DIM;
    int blocksPerGrid = (G->N+threadsPerBlock-1)/threadsPerBlock;
    PageRank_edge_centric(G,10000,df,blocksPerGrid,threadsPerBlock);
    cout<<"PageRank calculation done!!"<<endl;

    storePageRank(G,output_file);
    return 0;
}
