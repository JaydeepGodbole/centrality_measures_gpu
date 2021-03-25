#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

void addEdge(int *adj, int u, int v)
{
    adj[u*5+v] = 1;
    adj[v*5+u] = 1;
}

void printGraph(int *adj, int V)
{
    for (int v = 0; v < V; ++v)
    {
        cout << "\n Adjacency list of vertex "
             << v << "\n head ";
        for (int i = 0; i<V;i++)
        {   
            if(adj[v*V+i]==1)
            {
                cout << "-> " << i;
            }
        }
        printf("\n");
    }
}

__device__ struct ls
{   
    int value;
    ls *next=NULL;
};

__device__ ls *pushq(ls *q,int val)
{   
    ls *head = q;
    ls *temp = (ls*)malloc(sizeof(ls));
    
    if(q==NULL)
    {       
        
        //temp->value = val;
        q = (ls*)malloc(sizeof(ls));
        q->value = val;
        q->next = NULL;
        return q;
    }
    else
    {
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

__device__ int qfront(ls *q)
{   
    if(q!=NULL)
        return q->value;
    return -1;
}

__device__ ls *qpop(ls *q)
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


__device__ void bfs(int src, int dest, int *adj, int v,
         int *pred, int *dist, int *val)
{
    //list<int> queue;

    ls *queue = NULL;
    

    bool *visited;
    visited = (bool*)malloc(v*sizeof(bool));

    for (int i = 0; i < v; i++) {
        visited[i] = false;
        dist[i] = INT_MAX;
        pred[i] = -1;
    }
 
    visited[src] = true;
    dist[src] = 0;
    queue = pushq(queue,src);
    //printf("Hi \n");
    // standard BFS algorithm
    while (queue!=NULL) {
        int u = qfront(queue);
        queue = qpop(queue);
        for (int j = 0; j < v; j++) 
        {
            if (visited[adj[u*v+j]] == false) 
            {   
                visited[adj[u*v+j]] = true;
                dist[adj[u*v+j]] = dist[u] + 1;
                pred[adj[u*v+j]] = u;
                queue = pushq(queue,adj[u*v+j]);

                if (adj[u*v+j] == dest)
                {
                    *val = 1;
                    free(queue);
                    return;// true;
                }
            }
        }
        //break;
    }
 	*val = 0;
    free(visited);
    free(queue);
    return;// false;
}
 
__global__ void sd(int *adj, int *score, int v)
{
    int th = blockDim.x * blockIdx.x + threadIdx.x;
    
    // if(th<v)
    // {
    //     score[th] += 1; 
    // }
    //printf("%d\n", adj[0]);
    if(th<v)
    {
    	for(int j=0;j<v;j++)
    	{

		    int *pred, *dist;
            pred = (int*)malloc(v*sizeof(int));
            dist = (int*)malloc(v*sizeof(int));

		    int *val;
		    int x = 0;
		    val = &x;
            //printf("Hi\n");
		 	bfs(th, j, adj, v, pred, dist, val);
		    if (*val == 0) 
		    {
		        break;
		    }
		    // vector path stores the shortest path
		    int *path;
            path = (int*)malloc(v*sizeof(int));
            for(int i=0;i<v;i++)
            {
                path[i] = -1;
            }
		    int crawl = j;
		    path[0] = crawl;
            int pval = 1;
		    while (pred[crawl] != -1) {
		        path[pval] = pred[crawl];
                pval += 1;
		        crawl = pred[crawl];
		    }
		    int flag = 0;
		    for (int i = v-1; i >= 0; i--)
		    {
		        //cout << path[i] << " ";
                if(path[i]!=-1)
                {      
                    if(flag==0)
                    {
                        flag = 1;
                        continue;
                    }

                    score[path[i]] += 1;
                }
		    	
		    }
            free(path);
            free(pred);
            free(dist);
		}
	}

    //printf("Hi here %d\n", score[1]);
    //__syncthreads();

} 

int main()
{
    int V = 5;
    //vector<int> adj[V];
    int *adj;
    adj = (int*)malloc(V*V*sizeof(int));
    // for(int i=0;i<V;i++)
    // {
    //     adj[i] = (int*)malloc(V*sizeof(int));
    // }

    for(int i=0;i<V;i++)
    {
        for(int j=0;j<V;j++)
        {
            adj[i*V+j] = 0;
        }

    }    
 
    addEdge(adj, 0, 1);
    addEdge(adj, 0, 4);
    addEdge(adj, 1, 2);
    addEdge(adj, 1, 3);
    addEdge(adj, 1, 4);
    addEdge(adj, 2, 3);
    addEdge(adj, 3, 4);
    printGraph(adj, V);
    /* calculate betweenness centrality score using bfs */

    int score[V];
    for (int i = 0; i < V; ++i)
    {
    	score[i] = 0;
    }
   

 // 	for (int i = 0; i<V; i++)
 // 	{	
 // 		for(int j=0;j<V;j++)
 // 		{
 // 			sd(i,j,adj,V,score);
 // 		}
 // 	}   

 // 	for(int j=0;j<V;j++)
	// {
	// 	printf("The Score %f \n",(float(score[j])/(V*(V-1))));
	// }
 	
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t size = V*sizeof(float);//+ V*sizeof(adj[0][0]);
    //printf("size %d", size);
	
    //int **adj_A = (int **)malloc(V);
    // for(int i =0;i<V;i++)
    // {

    // }
    int *adj_A;

    err = cudaMalloc((void **)&adj_A,V*V*sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *score_A;
    err = cudaMalloc((void **)&score_A,V*sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device score A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(adj_A, adj, V*V*sizeof(int), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy adjacent matrix from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(score_A, score, V*sizeof(int), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy score from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    int threadsPerBlock = 5;
    int blocksPerGrid = 1;//(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    sd<<<blocksPerGrid, threadsPerBlock>>>(adj_A, score_A, V);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *result = (int*)malloc(V*sizeof(int));

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(result, score_A, V*sizeof(int), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    err = cudaFree(adj_A);
    err = cudaFree(score_A);


    for(int j=0;j<V;j++)
    {
     printf("The Score %f \n",(float(result[j])/(V*(V-1))));
    }

    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}