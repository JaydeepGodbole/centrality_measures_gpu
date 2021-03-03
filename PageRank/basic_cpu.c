#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float df = 0.85f;

typedef struct _Graph {
    int N; // number of nodes
    int **adj; // adjacency list
    int *outdegree; // contains outdegree of all nodes
    float *pr; // page rank values
} Graph;

// Finds max of 2 integers
int max(int a, int b)
{
    return (a>b)? a : b;
}

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
// Stores the outdegree of each node in array outdegree
// Returns the number of nodes in the graph
int edgeListToAdjacencyList(int **edges, int E, int ***adj, int **outdegree)
{
    int N = 0;
    for (int i=0; i<E; i++) {
        N = max(N, max(edges[i][0]+1, edges[i][1]+1));
    }

    *adj = (int**)malloc(N * sizeof(int*));
    *outdegree = (int*)malloc(N * sizeof(int));
    int* temp_outdegree = (int*)malloc(N * sizeof(int));

    for (int i=0; i<N; i++) {
        (*outdegree)[i] = 0;
        temp_outdegree[i] = 0;
    }
    for (int i=0; i<E; i++) {
        (*outdegree)[edges[i][0]]++;
        temp_outdegree[edges[i][0]]++;
    }

    for (int i=0; i<N; i++) {
        (*adj)[i] = (int*)malloc((*outdegree)[i] * sizeof(int));
    }
    for (int i=0; i<E; i++) {
        int u = edges[i][0];
        int v = edges[i][1];
        (*adj)[u][--temp_outdegree[u]] = v;
    }

    return N;
}

// Reads graph from input.txt stored as edge list and returns Graph structure
Graph readGraph()
{
    FILE *in_file = fopen("input.txt", "r");

    int E = 0;
    fscanf(in_file, "%d", &E);
    int** edges = readEdgeList(in_file, E);
    fclose(in_file);

    int** adj_list = NULL;
    int* out_list = NULL;

    Graph graph;
    graph.N = edgeListToAdjacencyList(edges, E, &adj_list, &out_list);
    graph.adj = adj_list;
    graph.outdegree = out_list;

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
void calculatePageRankOnce(Graph *graph) 
{
    float *old_pr = (float*)malloc(graph->N * sizeof(float));
    for (int i=0; i<graph->N; i++) {
        old_pr[i] = graph->pr[i];
    }

    for (int i=0; i<graph->N; i++) {
        graph->pr[i] = 0.0f;
        for (int j=0; j<graph->outdegree[i]; j++) {
            int w = graph->adj[i][j];
            graph->pr[i] += df * old_pr[w]/(float)graph->outdegree[w];
        }
        graph->pr[i] += (1-df)/(float)graph->N;
    }
    free(old_pr);
}   

// Calculates the page rank values of the given Graph structure for given number of iterations
void calculatePageRank(Graph *graph, int iter)
{
    while (iter--) {
        calculatePageRankOnce(graph);
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

    for (int i=0; i<graph->N; i++) {
        printf("%d -> ", i);
        for (int j=0; j<graph->outdegree[i]; j++) {
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
    calculatePageRank(&graph, 100);
    storePageRank(&graph);
}