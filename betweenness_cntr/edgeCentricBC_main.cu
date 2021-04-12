%%cuda --name main.cu

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <boost/algorithm/string.hpp>
#include </content/src/utils.h>
#include </content/src/kernels.h>

using namespace std;
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