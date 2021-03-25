#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float df = 0.85f;

typedef struct _Graph {
    int N; // number of nodes
    int count_sink; //number of sink nodes
    int **adj; // adjacency list of transposed graph
    int *outdegree; // contains outdegree of all nodes
    int *indegree; //contains indegree of all nodes
    int *sinks; //sink nodes
    float *pr; // page rank values
} Graph;

// Reads a 2D array of size (E*2) from the given file and returns it
int** readEdgeList(FILE *file, int E)
{
    int **edges = (int**)malloc(E * sizeof(int*));
    for (int i=0; i<E; i++) {
        edges[i] = (int*)malloc(2 * sizeof(int));
        fscanf(file, "%d %d", &edges[i][0], &edges[i][1]);
    }
    return edges;
}

// Builds adjacency list from edge list and stores it in adj
// Stores the outdegree and indegree of each node in arrays outdegree and indegree respectively
// Stores sink nodes in array sinks
void edgeListToAdjacencyList(int **edges, int E, int V, int *count_sink, int ***adj, int **outdegree, int **indegree, int **sinks)
{
    int N = V;

    *adj = (int**)malloc(N * sizeof(int*));
    *outdegree = (int*)malloc(N * sizeof(int));
    *indegree = (int*)malloc(N * sizeof(int));
    int* temp_indegree = (int*)malloc(N * sizeof(int));

    for (int i=0; i<N; i++) {
        (*outdegree)[i] = 0;
        (*indegree)[i] = 0;
        temp_indegree[i] = 0;
    }
    for (int i=0; i<E; i++) {
        (*outdegree)[edges[i][0]]++;
        (*indegree)[edges[i][1]]++;
        temp_indegree[edges[i][1]]++;
    }
 
    for (int i=0; i<N; i++){
        if ((*outdegree)[i] == 0){
            (*count_sink) += 1;
        }
    }
 
    *sinks = (int*)malloc((*count_sink)*sizeof(int));
  
    int x = 0;
    for (int i=0; i<N; i++){
        if ((*outdegree)[i] == 0){
            (*sinks)[x] = i;
            x+=1;
        }
    }

    for (int i=0; i<N; i++) {
        (*adj)[i] = (int*)malloc((*indegree)[i] * sizeof(int));
    }
    for (int i=0; i<E; i++) {
        int u = edges[i][0];
        int v = edges[i][1];
        (*adj)[v][--temp_indegree[v]] = u;
    }
    return;
}

// Reads graph from input.txt stored as edge list and returns Graph structure
Graph readGraph()
{
    FILE *in_file = fopen("input.txt", "r");

    int E,V = 0;
    fscanf(in_file, "%d %d", &E,&V);
    printf("Edges = %d, Vertices = %d\n", E, V);
    int** edges = readEdgeList(in_file, E);
    fclose(in_file);
 
    int** adj_list = NULL;
    int* out_list = NULL;
    int* in_list = NULL;
    int* sink_list = NULL;
    int sink_int = 0;

    Graph graph;
    graph.N = V;
    edgeListToAdjacencyList(edges, E, V, &sink_int, &adj_list, &out_list, &in_list, &sink_list);
    graph.count_sink = sink_int;
    graph.adj = adj_list;
    graph.outdegree = out_list;
    graph.indegree = in_list;
    graph.sinks = sink_list;

    return graph;
}

// Initialises the page rank values of the Graph structure to 1
void initialisePageRank(Graph *graph)
{
    graph->pr = (float*)malloc(graph->N * sizeof(float));
    for (int i=0; i<graph->N; i++) {
        graph->pr[i] = 1.0f;
        // graph->pr[i] = 1.0f/graph->N;
    }
}

// Calculates the page rank values of the given Graph structure for 1 iteration
float calculatePageRankOnce(Graph *graph) 
{
    float *old_pr = (float*)malloc(graph->N * sizeof(float));
    for (int i=0; i<graph->N; i++) {
        old_pr[i] = graph->pr[i];
    }

    for (int i=0; i<graph->N; i++) {
        graph->pr[i] = 0.0f;
        for (int j=0; j<graph->indegree[i]; j++) {
            int w = graph->adj[i][j];
            graph->pr[i] += df * old_pr[w]/(float)graph->outdegree[w];
        }

        //Add PageRank ontributed by all sink nodes
        for (int j=0; j<graph->count_sink; j++){
            graph->pr[i] += df * old_pr[graph->sinks[j]]/(float)graph->N;
        }
        graph->pr[i] += (1-df)/(float)graph->N;
    }

    float error = 0.0;
	for(int i=0; i<graph->N; i++){
	    error =  error + fabs(graph->pr[i] - old_pr[i]);
	}
    free(old_pr);
    return (error);
}   

// Calculates the page rank values of the given Graph structure for given number of iterations
void calculatePageRank(Graph *graph, int iter)
{
    float error = 0.0;
    while (iter--) {
        error = calculatePageRankOnce(graph);
        if (error < 0.000001){
	        break;
        }
    }
}

// Stores the page rank values of the given Graph structure in output.txt
void storePageRank(Graph *graph)
{
    FILE *out_file = fopen("output.txt", "w");
    for (int i=0; i<graph->N; i++) {
        fprintf(out_file, "%f\n", graph->pr[i]);
    }
    fclose(out_file);
}

// Prints number of nodes and adjacency list of given Graph structure in console
void printGraph(Graph *graph)
{
    printf("Nodes = %d\n", graph->N);
    printf("Sink Nodes = %d\n", graph->count_sink);

    for (int i=0; i<graph->N; i++) {
        printf("%d <- ", i);
        for (int j=0; j<graph->indegree[i]; j++) {
            printf("%d ", graph->adj[i][j]);
        }
        printf("\n");
    }
}

int main()
{
    Graph graph = readGraph();
    printGraph(&graph);
    initialisePageRank(&graph);
    calculatePageRank(&graph, 10000);
    storePageRank(&graph);
}
