#include <bits/stdc++.h>
//#include <cuda.h>
//#include <cuda_runtime.h>

using namespace std;

struct ls
{   
    int value;
    ls *next=NULL;
};

ls *pushq(ls *q,int val)
{   
    ls *head = q;
    ls *temp = (ls*)malloc(sizeof(ls));
    
    if(q==NULL)
    {       
        
        //temp->value = val;
        //cout<<"null val "<<flush;
        q = (ls*)malloc(sizeof(ls));
        q->value = val;
        q->next = NULL;
        return q;
    }
    else
    {   
        //cout<<"some val "<<flush;
        while(q->next!=NULL)
        {
            
            q = q->next;
        }
        
        temp->value = val;
        temp->next = NULL;
        q->next = temp;
        //q = q->next;

    }
    //printf("Queue value : %d",q->value);

    //free(temp);
    return head;// head;
}

int qfront(ls *q)
{   
    if(q!=NULL)
        return q->value;
    return -1;
}

ls *qpop(ls *q)
{
    if(q!=NULL)
    {
        ls *temp = (ls*)malloc(sizeof(ls));
        temp = q;
        q = q->next;
        free(temp);
        return q;
    }
    return q;
}


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

void bfs(int src, int dest, vector<int> adj[], int v,
         int pred[], int dist[], int *val)
{
    //list<int> queue;

    ls *queue = NULL;

    bool visited[v];

    for (int i = 0; i < v; i++) {
        visited[i] = false;
        dist[i] = INT_MAX;
        pred[i] = -1;
    }
 
    visited[src] = true;
    dist[src] = 0;
    //queue.push_back(src);
    int s = 0;
    //cout<<"Hi here"<<flush;
    queue = pushq(queue,src);
    //cout<<"Hi here2 "<<flush;
    s += 1;
    // standard BFS algorithm
    //cout<<"Hi here 3 "<<s<<flush;
    while (queue!=NULL) //!queue.empty()
    {   
        //cout<<"s : "<<s<<" "<<flush;
        int u = qfront(queue); //queue.front();
        //queue.pop_front();
        queue = qpop(queue);
        s -= 1;
        for (int i = 0; i < adj[u].size(); i++) 
        {   

            if (visited[adj[u][i]] == false) 
            {
                visited[adj[u][i]] = true;
                dist[adj[u][i]] = dist[u] + 1;
                pred[adj[u][i]] = u;
                //queue.push_back(adj[u][i]);
                //cout<<"seg here"<<flush;
                queue = pushq(queue,adj[u][i]);
                //cout<<"seg here af"<<flush;
                //cout<<queue->value;
                s += 1;
                if (adj[u][i] == dest)
                {   
                    //cout<<"if cond"<<flush;
                    *val = 1;
                    free(queue);
                    //cout<<"last"<<flush;
                    return;    
                }
                    
            }
        }
        //cout<<"s : "<<s<<" "<<flush;
    }
    //cout<<"s : "<<s<<" "<<flush;
 	*val = 0;
    free(queue);
    return;
}
 
void sd(int s,int dest, vector<int> adj[], int v, int *score)
{
    //int th = blockDim.x * blockIdx.x + threadIdx.x;
    

    int pred[v], dist[v];
    int *val;
    int x = 0;
    val = &x;
    //cout<<"last"<<flush;
    bfs(s, dest, adj, v, pred, dist,val);
    if(*val==0)
    {
        return;
    }  
    //cout<<"last"<<flush;
    //  vector path stores the shortest path
    vector<int> path;
    int crawl = dest;
    path.push_back(crawl);
    while (pred[crawl] != -1) {
        path.push_back(pred[crawl]);
        crawl = pred[crawl];
    }
    //cout<<path.size();
    for (int i = path.size() - 2; i >= 1; i--)
    {
        //cout << path[i] << " ";
        //cout<<"last"<<flush;
        score[path[i]] += 1;
    }
    //cout<<"last"<<flush;
    
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

    int *score;
    score = (int*)malloc(V*sizeof(int));
    for (int i = 0; i < V; ++i)
    {
    	score[i] = 0;
    }
   

 	for (int i = 0; i<V; i++)
 	{	
 		for(int j=0;j<V;j++)
 		{    
            //  cout<<"last "<<i<<" "<<j<<flush;
 			sd(i,j,adj,V,score);
            //cout<<"last "<<i<<" "<<j<<flush;
 		}
 	}   

 	for(int j=0;j<V;j++)
	{
		printf("The Score %f \n",(float(score[j])/(V*(V-1))));
	}

    return 0;
}