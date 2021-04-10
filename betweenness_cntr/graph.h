#include <bits/stdc++.h>
using namespace std;

class Graph {

public:

	int nodeCount, edgeCount;
	int *adjacencyList, *adjacencyListPointers;
	int *edgeList1, *edgeList2;

public:

	int getNodeCount() {
		return nodeCount;
	}

	int getEdgeCount() {
		return edgeCount;
	}

	// Reads the graph and stores in Compressed Sparse Row (CSR) format
	void readGraph() {

		// Reading in CSR format
		cout<<"Enter the nodes and edges count :\n";
		cin >> nodeCount >> edgeCount;

		// Copy into compressed adjacency List
		adjacencyListPointers = new int[nodeCount +1];
		adjacencyList = new int[2 * edgeCount +1];

		cout<<"Enter the row adjacency list:\n";

		for(int i=0; i<=nodeCount; i++) 
			cin >> adjacencyListPointers[i];

		cout<<"Enter the column adjacency list:\n";
		for(int i=0; i<(2 * edgeCount); i++)
			cin >> adjacencyList[i];
	}

	void convertToCOO() {
		edgeList2 = adjacencyList;
		edgeList1 = new int[2 * edgeCount +1];

		for(int i=0; i <nodeCount; ++i) {
			for(int j=adjacencyListPointers[i]; j<adjacencyListPointers[i+1]; ++j){
				edgeList1[j] = i;
			}
		}
	}

	int *getAdjacencyList() {

		return adjacencyList;
	}

	int *getAdjacencyListPointers() {

		return adjacencyListPointers;
	}
};