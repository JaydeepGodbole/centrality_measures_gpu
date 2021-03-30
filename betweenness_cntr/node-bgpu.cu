#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

void addEdge(int *adj, int u, int v, int t)
{
    adj[u*t+v] = 1;
    adj[v*t+u] = 1;
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
    ls *next;
};

__device__ ls *pushq(ls *q,int val)
{   
    ls *head = q;
    ls *temp = (ls*)malloc(sizeof(ls));
    
    if(q==NULL)
    {       

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

    }

    return head;
}

__device__ int qfront(ls *q)
{   
    if(q!=NULL)
        return q->value;
    return -1;
}

__device__ int qfrontt(ls *q)
{      
    if(q==NULL)
        return -1;
    while(q->next!=NULL)
    {
        q = q->next;
    }
    return q->value;
}
__device__ ls *qpop(ls *q)
{
    if(q!=NULL)
    {
        ls *temp;// = (ls*)malloc(sizeof(ls));
        temp = q;
        q = q->next;
        free(temp);
        return q;
    }
    return q;
}

__device__ ls *qpopt(ls *q)
{
    if(q!=NULL)
    {
        ls *temp,*head;// = (ls*)malloc(sizeof(ls));
        temp = q;
        head = q;
        ls *fr;
        fr = NULL;
        if(q->next==NULL)
            return NULL;
        while(q->next!=NULL)
        {   
            fr = q;
            q = q->next;
            temp = q;
        }
        fr->next = NULL;
        free(temp);
        return head;
    }
    return q;
}

__device__ void bfs(int src,const int *adj,const int v,
         ls **pred, int *dist, ls **visitstack, int *sigma, float *val)
{
    ls *queue = NULL;
    

    int *visited;
    visited = (int*)malloc(v*sizeof(int));
    if (visited == NULL) 
    {
        return;
    }

    // for (int i = 0; i < v; i++) {
    //     visited[i] = 0;
    //     dist[i] = 1000;
    //     pred[i] = -1;
    // }
    memset(visited, 0, v * sizeof(int));
    memset(dist, 1000, v * sizeof(int));
    //memset(pred, -1, v * sizeof(int));

    visited[src] = 1;
    dist[src] = 0;
    sigma[src] = 1;

    queue = pushq(queue,src);
    
    // standard BFS algorithm
    *val = 0;
    float closeness = 0;
    while (queue!=NULL) {
        int u = qfront(queue);
        queue = qpop(queue);
        *visitstack = pushq(*visitstack,u);
        //printf("Thread %d : queue %d \n",src,qfront(visitstack));
        //break;

        closeness += dist[u];
        for (int j = 0; j < v; j++) 
        {
            if ((visited[j] == 0) && (adj[u*v+j]==1)) 
            {   
                visited[j] = 1;
                dist[j] = dist[u] + 1;
                //pred[j] = u;
                queue = pushq(queue,j);

                // if (j == dest)
                // {
                //     *val = 1;
                //     free(queue);
                //     free(visited);
                //     return;
                // }
            }

            if (dist[j] == dist[u]+1)
            {
                pred[j] = pushq(pred[j],u);
                sigma[j] += sigma[u];
                // free(queue);
                // free(visited);
                //return;
            }
        }
        
    }
    
    if(closeness!=0)
    {
        *val = 1.0/closeness;
    }

    free(visited);
    free(queue);
    return;
}

__global__ void sd(int s,const int *adj, float *score,const int v)
{
    int th = blockDim.x * blockIdx.x + threadIdx.x ;

    if(th<v)
    {       

        //extern __shared__ int sc[];
        //sc[th] = score[th];
        __syncthreads();

        //printf("%d",score[th]);
    	for(int j=0;j<1;j++)
    	{     
            //if(th==249)
                //printf("%d %d\n", th, j);
            // if(j%10==0)
            // {
            //     printf("Hi %d \t %d \t",th,j);
            // }
            if(1)
            {   
                int *dist, *sigma;
                ls **pred;
                float *delta;

                pred = (ls**)malloc(v*sizeof(ls*));
                dist = (int*)malloc(v*sizeof(int));
                sigma = (int*)malloc(v*sizeof(int));
                delta = (float*)malloc(v*sizeof(float));

                if (sigma == NULL) 
                {
                    return;
                }
                if (pred == NULL) 
                {
                    return;
                }
                if (dist == NULL) 
                {
                    return;
                }
                memset(sigma, 0, v * sizeof(int));
                memset(delta, 0.0, v * sizeof(float));

                ls *visitstack=NULL;

                float *val;
                float x = 0;
                val = &x;
                

                
                bfs(th, adj, v, pred, dist, &visitstack, sigma, val);

                //printf("Thread %d : queue %d \n",th,qfront(visitstack));
                while (visitstack!=NULL) 
                {   

                    int w = qfrontt(visitstack);
                    visitstack = qpopt(visitstack);
                    //printf("%d ",w);
                    
                    // For each predecessors of node w, do the math!
                    while(pred[w]!=NULL) 
                    {   
                        int v = qfront(pred[w]);
                        pred[w] = qpop(pred[w]);
                        float c = ((float) sigma[v] / (float) sigma[w]) * (1.0 + delta[w]);
                        //printf("%f",c);
                        delta[v] += c;

                    }
                
                    // Node betweenness aggregation part.
                    if (w != th) 
                    {
                        //score[w] += delta[w];
                        atomicAdd(&score[w],delta[w]);
                    }
                }
                free(visitstack);
                free(pred);
                free(dist);
                free(delta);
            }
            
		    
		}

        
        // = sc[th];
        __syncthreads();
	}

} 

int main()
{
    int V = 250;
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
    //addEdge(adj, 0, 249, V);
    //printGraph(adj, V);
    /* calculate betweenness centrality score using bfs */

    float *score;
    score = (float*)malloc(V*sizeof(float));
    for (int i = 0; i < V; ++i)
    {
    	score[i] = 0;
    }
 	
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int *adj_A;
    float *result = (float*)malloc(V*sizeof(float));

    err = cudaMalloc((void **)&adj_A,V*V*sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *score_A;
    err = cudaMalloc((void **)&score_A,V*sizeof(float));

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


    int threadsPerBlock = 256;
    int blocksPerGrid = ((V-1)/256)+1;//(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    sd<<<blocksPerGrid, threadsPerBlock>>>(0,adj_A, score_A, V);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(result, score_A, V*sizeof(int), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(adj_A);
    err = cudaFree(score_A);


    for(int j=0;j<10;j++)
    {
     printf("The Score %f \n",(float(result[j])));
    }

    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(result);
    free(score);
    free(adj);


    return 0;
}