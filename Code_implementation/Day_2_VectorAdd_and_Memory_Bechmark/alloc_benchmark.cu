// alloc_benchmark.cu
//
// Measures average allocation+deallocation latency for:
//   1) pageable host memory via malloc()/free()
//   2) pinned host memory via cudaMallocHost()/cudaFreeHost()
//
// It uses C++ chrono steady_clock for wall‐clock timing.
//
// Compile with:
//   nvcc -arch=sm_89 -std=c++17 -O2 -o alloc_benchmark alloc_benchmark.cu
//
// Run:
//   ./alloc_benchmark <size_in_kb> <iterations>
// Example:
//   ./alloc_benchmark 1024 10000   # 1 MiB buffers, 10000 iterations

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// measure avg. time (microseconds) for malloc() + free()
double time_malloc_free(size_t bytes, int iters) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        void* p = malloc(bytes);
        free(p);
    }
    auto end = std::chrono::steady_clock::now();
    double total = std::chrono::duration<double, std::micro>(end - start).count();
    return total / iters;
}

// measure avg. time (microseconds) for cudaMallocHost() + cudaFreeHost()
double time_cudaMallocHost_freeHost(size_t bytes, int iters) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        void* p = nullptr;
        cudaMallocHost(&p, bytes);
        cudaFreeHost(p);
    }
    auto end = std::chrono::steady_clock::now();
    double total = std::chrono::duration<double, std::micro>(end - start).count();
    return total / iters;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <size_in_kb> <iterations>\n", argv[0]);
        return EXIT_FAILURE;
    }
    size_t kb = std::strtoul(argv[1], nullptr, 10);
    int iters = std::atoi(argv[2]);
    size_t bytes = kb * 1024;

    std::printf("Buffer size: %zu KB, Iterations: %d\n", kb, iters);

    double t_pageable = time_malloc_free(bytes, iters);
    std::printf(" malloc/free       : %8.3f microseconds per alloc+free\n", t_pageable);

    double t_pinned = time_cudaMallocHost_freeHost(bytes, iters);
    std::printf(" cudaMallocHost    : %8.3f microseconds per alloc+free\n", t_pinned);

    return EXIT_SUCCESS;
}
