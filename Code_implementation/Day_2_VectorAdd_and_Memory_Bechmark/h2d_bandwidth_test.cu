// h2d_bandwidth_test.cu
//
// This program measures H2D bandwidth for:
//   1) pageable host memory (malloc/free) and
//   2) pinned host memory (cudaMallocHost/cudaFreeHost)
//
// It times cudaMemcpyAsync() with CUDA events and prints GB/s for each case.
//
// Compile:
//   nvcc -arch=sm_89 -o h2d_bandwidth_test h2d_bandwidth_test.cu
//
// Run:
//   ./h2d_bandwidth_test [size_in_megabytes]
//   e.g. ./h2d_bandwidth_test 1024      # test with 1 GiB transfers

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static float measure_bandwidth(float* h_ptr, float* d_ptr, size_t bytes) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // record, copy, record, synchronize
    cudaEventRecord(start);
    cudaMemcpyAsync(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Convert to GB/s: (bytes / 1e9) / (ms / 1e3)
    return (bytes * 1e-9f) / (ms * 1e-3f);
}

int main(int argc, char** argv) {
    // test size: default 256 MiB
    size_t mb = (argc > 0) ? strtoul(argv[1], nullptr, 10) : 256;
    size_t bytes = mb * 1024 * 1024;

    // allocate device buffer
    float* d_buf = nullptr;
    if (cudaMalloc(&d_buf, bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_buf failed\n");
        return EXIT_FAILURE;
    }

    // 1) pageable host memory
    float* h_pageable = (float*)malloc(bytes);
    if (!h_pageable) {
        fprintf(stderr, "malloc failed\n");
        return EXIT_FAILURE;
    }

    // 2) pinned host memory
    float* h_pinned = nullptr;
    if (cudaMallocHost(&h_pinned, bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed\n");
        return EXIT_FAILURE;
    }

    // touch the memory to ensure pages are resident
    for (size_t i = 0; i < bytes/sizeof(float); i++) {
        h_pageable[i] = h_pinned[i] = 1.0f;
    }

    // measure
    float bw_pinned   = measure_bandwidth(h_pinned,   d_buf, bytes);
    float bw_pageable = measure_bandwidth(h_pageable, d_buf, bytes);
    

    // report
    printf("H2D bandwidth (pageable): %0.1f GB/s\n", bw_pageable);
    printf("H2D bandwidth (pinned)  : %0.1f GB/s\n", bw_pinned);

    // clean up
    cudaFree(d_buf);
    free(h_pageable);
    cudaFreeHost(h_pinned);

    return EXIT_SUCCESS;
}
