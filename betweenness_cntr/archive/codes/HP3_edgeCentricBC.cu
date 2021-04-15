#include <vector>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <boost/algorithm/string.hpp>

using namespace std;
namespace utils{
class graph
{
 public:
 int n;
 int m;
 int * edges;
 int * froms;
 int * scores;

 /*void print_edges()
 {
  for(int i = 0; i < n; i++)
  {
   std::cout << "Starting " << i;
   if (edges_to[i] == -1)
   {
    std::cout << 0 << std::endl;
   }
   else
   {
    std::cout << (edges_to[i]-edges_from[i] + 1) << std::endl;
   }    
    
  }
 }
 void print_scores_to_file(char * file);
 */
};

graph initialize(std::vector<std::vector<int>> connections)
{
 utils::graph g;
 g.m = connections.size(); // Number of edges
 g.n = connections[g.m-1][1] + 1; // Number of nodes in the graph

 g.edges = new int[2*g.m];
 g.froms = new int[2*g.m];
 g.scores = new int[g.n];

 int curr_edge_pointer = 0;
 /*
 for(int j = 0; j < 2*g.m; j++)
  {
    g.edges[j] = connections[j][1];
    g.froms[j] = connections[j][0];
    //curr_edge_pointer += 1;
  }*/
 
 for(int i = 0; i < g.n; i++)
 {
  bool isassigned = 0;
  for(int j = 0; j < g.m; j++)
  {
   if(connections[j][0]==i)
   {
    if(!isassigned)
    {
     isassigned = 1;
    }
    g.edges[curr_edge_pointer] = connections[j][1];
    g.froms[curr_edge_pointer] = connections[j][0];
    curr_edge_pointer += 1; 
   }
   
   else if(connections[j][1]==i)
   {
    if(!isassigned)
    {
     isassigned = 1;
    }
    g.edges[curr_edge_pointer] = connections[j][0];
    g.froms[curr_edge_pointer] = connections[j][1];
    curr_edge_pointer += 1; 
   }
   
  }
 }

 //std::cout << curr_edge_pointer/2 << std::endl;

 return g;
}

}

__global__ void getBC(const int * nhbrs, const int * froms, const int m, const int n, float * nodeBC, int * distance, int * numSPs, float * dependency, bool * predecessor)
{
  int nedge = m;
  int nnode = n;
  
  for (int nid = threadIdx.x; nid < nnode; nid += blockDim.x) 
    {
      nodeBC[nid] = 0.0;
    }
  __syncthreads();
 
 for (int source = 0; source < nnode; source++)
 {
  for (int eid = threadIdx.x; eid < 2 * nedge; eid += blockDim.x) 
  {
    int from = froms[eid];
    if(from == source)
    {
      numSPs[from] = 1;
      distance[from] = 0;  
    }
    else
    {
      numSPs[from] = 0;
      distance[from] = -1;  
    }        
    predecessor[eid] = false;
    dependency[from] = 0;
  }
  __syncthreads();
 
  __shared__ bool done;
 
  int d = 0;
  done = false;
  while (!done){
    __syncthreads();
    done = true;
    d++;
    __syncthreads();
    for(int eid = threadIdx.x; eid < 2 * nedge; eid += blockDim.x){
      int from = froms[eid];
      if(distance[from]==d){
        int nhbr = nhbrs[eid];
        int nhbrDist = distance[nhbr];
        if (nhbrDist == -1)
        {
          distance[nhbr] = d + 1;
          nhbrDist = d + 1;
          done = false;
        }
        if(nhbrDist < d)
        {
          predecessor[eid] = true;
        }
        if(nhbrDist == d + 1)
        {
          atomicAdd(&numSPs[nhbr], numSPs[from]);
        }
      }
     
    }
    __syncthreads();
  }
  __syncthreads();

  while (d > 1){
    for (int eid = threadIdx.x; eid < 2 * nedge; eid += blockDim.x) 
    {
      int from = froms[eid];
      if(distance[from] == d)
      {
        if (predecessor[eid])
        {
          int nhbr = nhbrs[eid];
          float delta = (1.0 + dependency[from]) * (numSPs[nhbr] / numSPs[from]);
          atomicAdd(&dependency[nhbr], delta);
        }
      }
    }
    d--;
    __syncthreads();
  }
  
  __syncthreads();
 
  for (int nid = threadIdx.x; nid < nnode; nid += blockDim.x){
    nodeBC[nid] = nodeBC[nid] + dependency[nid];
  }
  __syncthreads();
}
}

float * edge_centric_bc_gpu(utils::graph g)
{
 cudaDeviceProp prop;
 cudaError_t err = cudaSuccess;
 err = cudaGetDeviceProperties(&prop, 0);
 if(err != cudaSuccess)
 {
  std::cout << "Failed" << std::endl;
  exit(EXIT_FAILURE);
 }
 std::cout << "Chosen Device: " << prop.name << std::endl;
 std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
 std::cout << "Number of Streaming Multiprocessors: " << prop.multiProcessorCount << std::endl;
 std::cout << "Size of Global Memory: " << prop.totalGlobalMem/(float)(1024*1024*1024) << " GB"<< std::endl;

 int max_threads_per_block = prop.maxThreadsPerBlock;
 int num_SMs = prop.multiProcessorCount;
 
 int * d_edges = NULL;
 int * d_froms = NULL;
 float * d_bc = NULL;
 float * d_bc_local = NULL;
 int * dist = NULL;
 int * num_shortest_paths = NULL;
 bool * predecessor = NULL;
 

 size_t size = g.n*sizeof(int);
 err = cudaMalloc((void**)&dist, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMalloc((void**)&num_shortest_paths, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 
 size = g.n*sizeof(float);
 err = cudaMalloc((void**)&d_bc, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMalloc((void**)&d_bc_local, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 
 size = 2*g.m*sizeof(int);
 err = cudaMalloc((void**)&d_edges, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMemcpy(d_edges, g.edges, size, cudaMemcpyHostToDevice);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMalloc((void**)&d_froms, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 err = cudaMemcpy(d_froms, g.froms, size, cudaMemcpyHostToDevice);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}

 size = 2*g.m*sizeof(bool);
 err = cudaMalloc((void**)&predecessor, size);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}
 
 cudaEvent_t start, stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);

 cudaEventRecord(start);
 getBC<<<num_SMs, max_threads_per_block>>>(d_edges, d_froms, g.m, g.n, d_bc, dist, num_shortest_paths, d_bc_local, predecessor);
 cudaEventRecord(stop);
 cudaEventSynchronize(stop);
 float milliseconds = 0;
 cudaEventElapsedTime(&milliseconds, start, stop);
 
 std::cout << "The running time is " << milliseconds << "milliseconds" << std::endl;
 
 float * bc_calculated = new float[g.n];

 for(int i = 0; i < g.n; i++)
 {
  bc_calculated[i] = 1.0;
 }

 size = g.n*sizeof(float);
 err = cudaMemcpy(bc_calculated, d_bc, size, cudaMemcpyDeviceToHost);
 if(err!=cudaSuccess){std::cout<<cudaGetErrorString(err)<<std::endl; exit(EXIT_FAILURE);}

 return bc_calculated;

}

class graph
{
 // Defining the graph data structure for handling large graphs
 public:
 int n; // Number of nodes in the graph
 int m; // Number of edges in the graph
 int * edges;
 int * froms;
 int * scores;

};
int main(int argc, char** argv)
{
 // First argument is the graph name (relative to the main directory).

 std::string s = argv[1];

 if(s.find(".txt") != std::string::npos)
  {
		;//return 0;//generate_graph(file);
	}
	else if(s.find(".edge") != std::string::npos)
	{
		;//return 0;//generate_graph(file);
	}
	else
	{
		std::cerr << "Error: Unsupported file type." << std::endl;
		exit(-1);
	}
 std::string datapath = "drive/MyDrive/HP3/Betweenness_Centrality/dataset/"; 
 s = datapath + s;
 std::cout << s << std::endl;
 std::fstream fin(s);
 std::string line;
 std::fstream graph_file;
 graph_file.open(s, ios::in);
 if(!graph_file)
 {
  cout << "File doesn't exist" << endl;
 }
 else
 {
  cout << "Graph file exists" << endl;
 }
 vector<vector<int>> edges;
 vector<int> temp;
 while (getline(fin, line)) {
  temp.clear();
  // Split line into tab-separated parts
  std::vector<std::string> parts;
  boost::algorithm::split(parts, line, boost::is_any_of(" "));
  // TODO Your code goes here!
  for(int i = 0; i < parts.size(); i++)
  {
   stringstream s(parts[i]);
   int k;
   s >> k;
   temp.push_back(k);
  }
  if(temp.size()!=2){
      cerr << "Format incorrect" << endl;
      return -1;
  }
 edges.push_back(temp);
 }
 fin.close();
 cout << "Number of edges in the graph: " << edges.size() << endl;

 utils::graph g = utils::initialize(edges);

 float * bc = edge_centric_bc_gpu(g);

 for(int i = 0; i < g.n; i++)
 {
  g.scores[i] = bc[i];
  std::cout << bc[i] << std::endl;
 }
 
 return 0; 
}