#include <bits/stdc++.h>

using namespace std;

void addEdge(vector<int> adj[], int u, int v)
{
    adj[u].push_back(v);
    adj[v].push_back(u);
}

void printGraph(vector<int> adj[], int V)
{
    for (int v = 0; v < V; ++v)
    {
        cout << "\n Adjacency list of vertex "
             << v << "\n head ";
        for (int i = 0; i<adj[v].size();i++)
        {
           cout << "-> " << adj[v][i];
        }
        printf("\n");
    }
}

void bfs(int u) 
{ 
    queue<int> q; 
  
    q.push(u); 
    v[u] = true; 
  
    while (!q.empty()) { 
  
        int f = q.front(); 
        q.pop(); 
  
        cout << f << " "; 
  
        // Enqueue all adjacent of f and mark them visited  
        for (auto i = g[f].begin(); i != g[f].end(); i++) { 
            if (!v[*i]) { 
                q.push(*i); 
                v[*i] = true; 
            } 
        } 
    } 
} 

int main()
{
    int V = 5;
    vector<int> adj[V];
    addEdge(adj, 0, 1);
    addEdge(adj, 0, 4);
    addEdge(adj, 1, 2);
    addEdge(adj, 1, 3);
    addEdge(adj, 1, 4);
    addEdge(adj, 2, 3);
    addEdge(adj, 3, 4);
    printGraph(adj, V);

    /* calculate betweenness centrality score using bfs */

    vector<int> score,bt_score;
    for (int i = 0; i < V; ++i)
    {
    	score.push_back(0);
    }
   

 	for (int i = 0; i<V; i++)
 	{	
 		for(int j=0;j<V;j++)
 		{
 			bfs(i,j,adj,score);
 		}
 	}   

 	printf("The Score %d",score[0]);

    return 0;
}