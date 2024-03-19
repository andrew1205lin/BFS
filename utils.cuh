#pragma once
#include "graph.h"
#include <chrono>
#include <cuda.h>
#include <iomanip> // put_time
#include <iostream>
#include <thread>

#define CHECK(x)                                                               \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__,                    \
              cudaGetErrorName(err), cudaGetErrorString(err));                 \
      exit(1);                                                                 \
    }                                                                          \
  } while (0);

#define DUMP(x)                                                                \
  do {                                                                         \
    std::cout << #x << ": " << x << std::endl;                                 \
  } while (0);

class CudaTimer {
public:
  CudaTimer() {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
  }

  ~CudaTimer() {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
  }

  void start() { cudaEventRecord(startEvent, 0); }

  void stop() {
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
  }

  float elapsed() const {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    return milliseconds;
  }

private:
  cudaEvent_t startEvent, stopEvent;
};

void print_duration(const std::chrono::steady_clock::duration &duration) {
  // convert the duration to floating-point seconds first
  double seconds = std::chrono::duration<double>(duration).count();

  // adaptively select the unit based on the duration
  if (seconds >= 1.0) {
    std::cout << std::setprecision(3) << seconds << " s" << std::endl;
  } else if (seconds >= 1e-3) {
    std::cout << std::setprecision(3) << (seconds * 1e3) << " ms" << std::endl;
  } else if (seconds >= 1e-6) {
    std::cout << std::setprecision(3) << (seconds * 1e6) << " us"
              << std::endl; // microseconds
  } else {
    std::cout << std::setprecision(3) << (seconds * 1e9) << " ns"
              << std::endl; // nanoseconds
  }
}
