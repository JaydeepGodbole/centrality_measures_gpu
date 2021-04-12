# centrality_measures_gpu
Repository for the term project of Group 5 for the course on High Performance Parallel Programming (CS61064) 2021

The code is supposed to implement algorithms, namely Pagerank and Betweenness centrality, on large graphs


## Betweenness Centrality
For Executing work efficient and edge centric Betweenness Centrality code, please follow the steps below:
1. Login to the Gmail Account with the following credentials:
   emailID: **eightsemprojects@gmail.com**
   password: **8thsemprojects**
2. Go to HP3 folder
3. Go to Betweenness_Centrality folder
4. Change Runtime type to GPU
5. Open work_efficient.ipynb and click on Run All to run work efficient betweenness centrality code
6. Open HP3_EdgeCentricBC_Group5.ipynb and click on Run All to run work efficient betweenness centrality code
7. Please note in both 5 and 6, you need to mount the drive. All the folder hierarchies are already maintained.
8. All the required packages have been already included in the .ipynb file
9. This should be enough to run both the codes.


## PageRank
 - Kernels
   - NodeCentric.h - kernel for pagerank calculation using NodeCentric approach
   - EdgeCentric_phase1.h - kernel-1 for pagerank calculation using EdgCentric approach
   - EdgeCentric_phase2.h - kernel-2 for pagerank calculation using EdgeCentric approach
   - EdgeCentric_phase1_optimized.h - optimized version of kernel-1 of EdgeCentric approach
 - Dataset
   - amazon.txt
   - facebook.txt
 - Profiling - profiling report of all the 4 kernels on amazon dataset
 - node_centric.cu - Host code for PageRank using NodeCentric approach
 - edge_centric.cu - Host code for PageRank using EdgeCentric approach
 - edge_centric_optimized.cu - Host code for WorkEfficient implementation of PageRank using EdgeCentric
 
 - ### steps to execute
   1. set the path to input file and output file in `input_file[]` & `output_file[]` array in `main()` function
   2. `nvcc node_centric.cu -o rank.out`
   3. `./rank.out`