// Day 2 – vectorAdd_global.cu (Basic Version)

// Compile with: nvcc -arch=sm_89 -o vectorAdd_global_basic vectorAdd_global_fixed.cu
// Run with: ./vectorAdd_global_basic

// This is a basic version of vector addition using CUDA
// It performs vector addition on two arrays A and B, storing the result in C.
// The arrays are allocated on the device and the result is copied back to the host.
// This code is intended for educational purposes and may require adjustments
// based on your specific CUDA setup and GPU capabilities.


#include <cstdio>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1 << 20;                  // 1M elements
    const size_t bytes = N * sizeof(float);

    // Host allocations
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    // Initialize
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2.0f * i;
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernel: 256 threads per block
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // Copy result back
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Quick check
    printf("C[0]=%f, C[N-1]=%f\n", h_C[0], h_C[N-1]);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
