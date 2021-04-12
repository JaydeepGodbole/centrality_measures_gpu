%%cuda --name utils.h
#ifndef UTILS
#define UTILS

#include <vector>
#include <cstdlib>
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
#endif