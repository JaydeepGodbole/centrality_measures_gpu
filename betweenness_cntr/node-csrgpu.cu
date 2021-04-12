#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernels/node-centric_kernels.h"

using namespace std;

int main(int argc, char *argv[])
{   
    if(argc<=1)
    {
        cout<<"Give csr file name as argument\n";
        return;
    }
    int V,E;

    /* calculate betweenness centrality score using bfs */
    
    Graph g;

    g.readGraphfile(argv[1]);  //facebook_combined
    
    double *result = betweenness_node(g);

    for (int i = 0; i < g.getNodeCount() ; ++i)
    {
        cout<<"Score :"<<result[i]<<endl;
    }

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