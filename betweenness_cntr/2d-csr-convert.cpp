#include <bits/stdc++.h>

using namespace std;

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

void convert_csr(int *adj, int *adjr, int *adjc, int v)
{   
    int sum = 0;
    for(int i=0;i<v;i++)
    {   
        for(int j=0;j<v;j++)
        {
            if(adj[i*v+j]==1)
            {   
                adjc[sum] = j;
                sum += 1;
            }    
        }
        adjr[i+1] = sum;
    }


}

int main()
{
    int V = 5;
    int *adj;
    adj = (int*)malloc(V*V*sizeof(int));

    for(int i=0;i<V;i++)
    {
        for(int j=0;j<V;j++)
        {
            adj[i*V+j] = 0;
        }

    }    
 
    addEdge(adj, 0, 1, V);
    addEdge(adj, 0, 4, V);
    addEdge(adj, 1, 2, V);
    addEdge(adj, 1, 3, V);
    addEdge(adj, 1, 4, V);
    addEdge(adj, 2, 3, V);
    addEdge(adj, 3, 4, V);
    
    int E = 7;

    int *adjr, *adjc;
    adjr = new int[V+1];
    adjc = new int[2*E+1];

    convert_csr(adj, adjr, adjc, V);

    printGraph(adjr, adjc, V);

    ofstream myfile;
    char *filename = "graph_csr.txt";
    myfile.open (filename);

    myfile<<V<<endl;
    myfile<<E<<endl;

    for(int i=0;i<V+1;i++)
    {
        myfile<<adjr[i]<<" ";
    }
    myfile<<endl;
    for(int i=0;i<2*E+1;i++)
    {
        myfile<<adjc[i]<<" ";
    }

    myfile.close();


    // ifstream file;
    // file.open(filename);
    // int nodeCount,edgeCount;

    // file >> nodeCount >> edgeCount;

    // // Copy into compressed adjacency List
    // int *adjacencyListPointers = new int[nodeCount +1];
    // int *adjacencyList = new int[2 * edgeCount +1];

    

    // for(int i=0; i<=nodeCount; i++) 
    //     file >> adjacencyListPointers[i];

    
    // for(int i=0; i<(2 * edgeCount); i++)
    //     file >> adjacencyList[i];

    // file.close();
    // printGraph(adjacencyListPointers, adjacencyList, V);

    return 0;
}