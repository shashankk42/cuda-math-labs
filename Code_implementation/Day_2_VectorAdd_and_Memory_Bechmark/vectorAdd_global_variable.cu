// Day 2 – vectorAdd_global_variable.cu 

// nvcc -arch=sm_89 -o vectorAdd_global vectorAdd_global_variable.cu

// # default N=1M:
// ./vectorAdd_global

// # or specify size, e.g. 8M elements:
// ./vectorAdd_global 8388608

// This code performs vector addition using CUDA with global variables.
// It initializes two arrays A and B on the host, allocates memory on the device,
// and performs the addition in a kernel. The result is copied back to the host.


// Uses pinned host memory (cudaMallocHost) for higher H2D bandwidth.
// Asynchronous copies and a kernel timed via CUDA events.
// Inline error checks without macros.
// Runtime-configurable problem size (via argv[1]).


#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't access out of bounds
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char* argv[]) {
    // Allow custom size via argv or default to 1M
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 20);
    size_t bytes = size_t(N) * sizeof(float);

    // Pinned host allocations for higher Host to Device bandwidth
    float *h_A, *h_B, *h_C;
    cudaError_t err = cudaMallocHost(&h_A, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "Host alloc A failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }
    err = cudaMallocHost(&h_B, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "Host alloc B failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }
    err = cudaMallocHost(&h_C, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "Host alloc C failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    // Initialize
    for (int i = 0; i < N; i++) {
        h_A[i] = float(i);
        h_B[i] = float(2 * i);
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    err = cudaMalloc(&d_A, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_A failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }
    err = cudaMalloc(&d_B, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_B failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }
    err = cudaMalloc(&d_C, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_C failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start
    cudaEventRecord(start);

    // Async copy host to device
    err = cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "H2D A failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }
    err = cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "H2D B failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    // Kernel launch
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    // Async copy device to host
    err = cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "D2H C failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    // Record stop & synchronize
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Bandwidth: 2 transfers of N floats (A and B) plus 1 transfer of N floats (C)
    double gb = double(bytes) * 3.0 / (1<<30);
    printf("VectorAdd (N=%d): Time = %.3f ms, BW = %.1f GB/s\n", N, ms, gb / (ms/1000.0));

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return EXIT_SUCCESS;
}
