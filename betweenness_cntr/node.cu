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

bool bfs(int src, int dest, vector<int> adj[], int v,
         int pred[], int dist[])
{
    // a queue to maintain queue of vertices whose
    // adjacency list is to be scanned as per normal
    // DFS algorithm
    list<int> queue;
 
    // boolean array visited[] which stores the
    // information whether ith vertex is reached
    // at least once in the Breadth first search
    bool visited[v];
 
    // initially all vertices are unvisited
    // so v[i] for all i is false
    // and as no path is yet constructed
    // dist[i] for all i set to infinity
    for (int i = 0; i < v; i++) {
        visited[i] = false;
        dist[i] = INT_MAX;
        pred[i] = -1;
    }
 
    // now source is first to be visited and
    // distance from source to itself should be 0
    visited[src] = true;
    dist[src] = 0;
    queue.push_back(src);
 
    // standard BFS algorithm
    while (!queue.empty()) {
        int u = queue.front();
        queue.pop_front();
        for (int i = 0; i < adj[u].size(); i++) {
            if (visited[adj[u][i]] == false) {
                visited[adj[u][i]] = true;
                dist[adj[u][i]] = dist[u] + 1;
                pred[adj[u][i]] = u;
                queue.push_back(adj[u][i]);
 
                // We stop BFS when we find
                // destination.
                if (adj[u][i] == dest)
                    return true;
            }
        }
    }
 
    return false;
}
 
// utility function to print the shortest distance
// between source vertex and destination vertex
vector<int> sd(int s,int dest, vector<int> adj[], int v, vector<int> score)
{
    // predecessor[i] array stores predecessor of
    // i and distance array stores distance of i
    // from s
    int pred[v], dist[v];
 
    if (bfs(s, dest, adj, v, pred, dist) == false) {
        //cout << "Given source and destination"
        //     << " are not connected";
        return score;
    }
 
    // vector path stores the shortest path
    vector<int> path;
    int crawl = dest;
    path.push_back(crawl);
    while (pred[crawl] != -1) {
        path.push_back(pred[crawl]);
        crawl = pred[crawl];
    }
 
    // distance from source is in distance array
    //cout << "Shortest path length is : "
    //     << dist[dest];
 
    // printing path from source to destination
    //cout << "\nPath is::\n";
    for (int i = path.size() - 2; i >= 1; i--)
    {
        //cout << path[i] << " ";
    	score[path[i]] += 1;
    }
    return score;
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
 			score = sd(i,j,adj,V,score);
 		}
 	}   

 	for(int j=0;j<V;j++)
	{
		printf("The Score %f \n",(float(score[j])/(V*(V-1))));
	}
 	
	
    return 0;
}