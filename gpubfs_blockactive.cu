#include <iostream>
#include "graph.h"
#include "color.h"
#include "utils.cuh"
#include <cassert>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

namespace cg = cooperative_groups;

constexpr int NUM_BFS = 10000;
constexpr int BLOCK_SIZE = 32;
constexpr int NUM_BLOCKS = 512;

template<typename T>
__device__ void setArray(T* array, T value, int array_len) {
    int lane = threadIdx.x + blockIdx.x * blockDim.x;
    for(int u=lane;u<array_len;u+=gridDim.x){
        array[u] = value;
    }
}

__global__ void cudaBFS(int num_nodes, const int* offsets, 
                                const int* destinations,
                                int* Fa, int* Fa_len, int* Sa,
                                int* source, int* sink, int* distance)
{
  /*
  Fa: Frontier id queue
  Xa: Visited array (len: #vertex) 
  Ca: Cost array    (len: #vertex)
  */
  cg::grid_group grid = cg::this_grid();
  int tid = threadIdx.x;//+ blockIdx.x * blockDim.x;
  int bid = blockIdx.x;
  
  for(int u=0;u<NUM_BFS;u++){ // choose source u
    //if(tid == 0) printf("source[u] = %d\n", source[u]);
    // initialization for BFS in this round 
    setArray(Fa, 0, num_nodes);
    setArray(Sa, -1, num_nodes);
    grid.sync();
    if(tid == 0 && bid == 0){
      *Fa_len = 0;
      Sa[source[u]] = 0;
      Fa[*Fa_len] = source[u];
      atomicAdd(Fa_len, 1);
    } 
    grid.sync();

    // do until Fa_len==0;
    int level=0;
    while(*Fa_len!=0){
      //if(tid == 0) printf("*Fa_len = %d\n", *Fa_len);
      // update status array with Vs in Frontier queue 
      for(int v=bid;v<*Fa_len;v+=gridDim.x){
        __shared__ int frontier;
        __shared__ int start;
        __shared__ int end;
        if(tid == 0){
          frontier = Fa[v];
          start = offsets[frontier];
          end = offsets[frontier+1];          
        }
        //printf("tid %d: frontier %d\n", tid, frontier);
        __syncthreads();

        for(int w_idx=start+tid;w_idx<end;w_idx+=blockDim.x){ 
          int w=destinations[w_idx];
          if(Sa[w]==-1) Sa[w]=level+1;
          //printf("tid %d: w=%d, Sa[w] = %d\n", tid, w, Sa[w]);
        }
        __syncthreads();
      }
      grid.sync(); //make sure status array are updated
      if(tid == 0 && bid == 0) *Fa_len=0;
      grid.sync(); //make sure Fa_len are updated

      // Scan the status array and generate new frontier queue
      
      for(int v=tid+blockDim.x*bid;v<num_nodes;v+=blockDim.x*gridDim.x){
        if(Sa[v] == level+1){
          //printf("tid %d: v=%d, Sa[v] = %d\n", tid, v, Sa[v]);
          int index = atomicAdd(Fa_len, 1);
          Fa[index] = v;
        } 
      }
      
      /*for(int v=tid;v<*Fa_len;v+=blockDim.x){
        printf("tid %d: v=%d, Fa[v] = %d\n", tid, v, Fa[v]);
      }*/
      
      level+=1;
      //if(tid == 0) printf("level %d\n", level);
      grid.sync(); //?
    }
    if(tid == 0 && bid == 0){
      sink[u] = Fa[0];
      distance[u] = level-1;
    }
    grid.sync();// if this line is miss, another BFS u would start too soon
  }

  /*for(int v=tid;v<NUM_BFS;v+=blockDim.x){
    printf("source = %d, sink = %d, distance = %d\n", source[v], sink[v], distance[v]);
  }*/

  return ;
}

/*__global__ void my_kernel(int* result)
{
    __shared__ int sum;
    sum = 0;
    
    for (int i = threadIdx.x; i < 10; i += blockDim.x)
    { 
      atomicAdd(&sum, i);
    }
    
    // 同步所有 block 中的 thread
    cg::grid_group grid = cg::this_grid();
    grid.sync();

    if (threadIdx.x == 0)
    {
        
        atomicAdd(result, sum);
    }
}*/

int main(int argc, char **argv) {
  // start_vertex~end_vertex
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <csr_binary_file> <out_file_path> <start_vertex> <end_vertex>" << std::endl;
    return 1;
  }

  // read file and save to csr
  std::string txt_file = std::string(argv[1]);
  std::string csr_binary_file = txt_file + ".bin";
  CSRGraph csr;
  if (csr.loadFromBinary(csr_binary_file)) {
    std::cout << "Loading " << txt_file
            << " from binary file..." << std::endl;
  } else {
    std::cout << "No binary file found, loading from txt file..." << std::endl;
    csr = std::move(CSRGraph(argv[1]));
    csr.checkIfContinuous();
    //csr.checkIfLowerTriangle();
    csr.saveToBinary(csr_binary_file);
    CSRGraph graph2;
    graph2.loadFromBinary(csr_binary_file);
    assert(csr == graph2);
  }

  //  should be equal to 1000
  int start_vertex;
  int end_vertex;
  int DATASET_VERTEX=1134890;

  for (start_vertex = std::stoi(argv[3]); start_vertex < DATASET_VERTEX;
       start_vertex += 10000) {
    end_vertex = start_vertex + NUM_BFS - 1;
    std::cout << "On-going.." << int((start_vertex / float(DATASET_VERTEX)) * 100.0)
              << " %\r";
    std::cout.flush();

    // allocate GPU memory
    int* d_offsets;
    int* d_destinations;

    int* d_frontier_queue;
    int* d_queue_length;
    int* d_status_array; // or distance_array

    int* d_source;
    int* d_sink;
    int* d_distance;
    std::vector<int> h_source_vector(NUM_BFS);
    for (int i = 0; i < NUM_BFS; ++i) {
      h_source_vector[i] = start_vertex + i;
    }


    CHECK(cudaMalloc(&d_offsets, csr.offsets.size() * sizeof(int)));
    CHECK(cudaMemcpy(d_offsets, csr.offsets.data(), 
                    csr.offsets.size() * sizeof(int), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&d_destinations, csr.destinations.size() * sizeof(int)));
    CHECK(cudaMemcpy(d_destinations, csr.destinations.data(),
                    csr.destinations.size() * sizeof(int),
                    cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&d_frontier_queue, sizeof(int) * csr.num_nodes));
    CHECK(cudaMalloc(&d_queue_length, sizeof(int)));
    CHECK(cudaMalloc(&d_status_array, sizeof(int) * csr.num_nodes));

    CHECK(cudaMalloc(&d_source, sizeof(int) * NUM_BFS));
    CHECK(cudaMalloc(&d_sink, sizeof(int) * NUM_BFS));
    CHECK(cudaMalloc(&d_distance, sizeof(int) * NUM_BFS));
    CHECK(cudaMemcpy(d_source, h_source_vector.data(), 
                    NUM_BFS*sizeof(int), cudaMemcpyHostToDevice));


    // run the kernel
    float totalMilliseconds = 0.0f;
    CudaTimer timer;

    dim3 grid(NUM_BLOCKS);
    dim3 block(BLOCK_SIZE);
    void* args[] = {&csr.num_nodes, &d_offsets, &d_destinations, 
          &d_frontier_queue, &d_queue_length, &d_status_array,
          &d_source, &d_sink, &d_distance};

    timer.start();
    /*cudaBFS<<<NUM_BLOCKS, BLOCK_SIZE>>>(
          csr.num_nodes, d_offsets, d_destinations,
          d_frontier_queue, d_queue_length, d_status_array,
          d_source, d_sink, d_distance);*/
    cudaLaunchCooperativeKernel((void*)cudaBFS, grid, block, 
          args);
    timer.stop();
    
    CHECK(cudaGetLastError());
    std::cout << "cudaBFS Kernel execution time: " << Color::kYellow
              << timer.elapsed() << " ms\n";
    totalMilliseconds += timer.elapsed();

    CHECK(cudaDeviceSynchronize());


    // copy the result from GPU
    int h_source[NUM_BFS];
    int h_sink[NUM_BFS];
    int h_distance[NUM_BFS];
    CHECK(cudaMemcpy(&h_source, d_source, sizeof(int) * NUM_BFS, 
                    cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&h_sink, d_sink, sizeof(int) * NUM_BFS, 
                    cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&h_distance, d_distance, sizeof(int) * NUM_BFS, 
                    cudaMemcpyDeviceToHost));

    // free GPU memory
    cudaFree(d_destinations);
    cudaFree(d_offsets);
    cudaFree(d_frontier_queue);
    cudaFree(d_queue_length);
    cudaFree(d_status_array);
    cudaFree(d_source);
    cudaFree(d_sink);
    cudaFree(d_distance);

    // write the result
    std::string filename = std::string(argv[2]) + "/output_" + std::to_string(start_vertex) + "_" + std::to_string(end_vertex) + ".txt";
    std::ofstream outfile(filename);
    if (!outfile.is_open()){
      std::cerr << "Error opening output file." << std::endl;
      return 1;
    }

    for(int i=0;i<NUM_BFS;i++){
      outfile << "source = " << h_source[i] 
              << ", sink = " << h_sink[i] 
              << ", distance = " << h_distance[i] 
              << std::endl;
    }

    outfile.close();  // Close the file
    

    /*
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    // initialize, then launch
    printf("deviceProp.multiProcessorCount: %d", deviceProp.multiProcessorCount);
    dim3 grid(deviceProp.multiProcessorCount);
    dim3 block(BLOCK_SIZE);

    int result = 0;
    int* d_result;

    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);

    void* args[] = {&d_result};
    cudaLaunchCooperativeKernel((void*)my_kernel, grid, block, args);

    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Result: " << result << std::endl;

    cudaFree(d_result);
    */
  }
  return 0;
}