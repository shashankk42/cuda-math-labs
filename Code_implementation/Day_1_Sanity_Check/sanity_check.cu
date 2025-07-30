// Day 1 – sanity_check.cu
// Check environment and device capabilities
// This code checks the CUDA environment, queries device capabilities,
// and launches a simple kernel to print block and thread IDs.

// Ensure you have the NVIDIA CUDA Toolkit installed and a compatible GPU.

// This code is intended for educational purposes and may require adjustments
// based on your specific CUDA setup and GPU capabilities.

// Compile with: nvcc -arch=sm_89 -run sanity_check.cu


#include <cstdio>           // For printf
#include <cstdlib>          // For exit()
#include <cuda_runtime.h>   
// Device queries and configuration 
// Memory management (cudaMalloc, cudaFree)
// Kernel launches (<<<…>>> syntax is enabled by nvcc)
// Synchronization and error checking


// Simple error‐checking macro
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr,                                                      \
                    "CUDA error at %s:%d – %s\n",                                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

//Kernel: Each thread prints its block and thread ID
__global__ void sanityCheckKernel() {
    printf("hello from block %d and thread %d \n",
           blockIdx.x, threadIdx.x);
}


int main() {
    // Query devices
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return EXIT_FAILURE;
    }

    // Print properties and check for Tensor Core support
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n",
               prop.major, prop.minor);
        int cc = prop.major * 10 + prop.minor;
        if (cc >= 70) {
            printf("  Tensor Cores: available (CC %d.%d)\n", prop.major, prop.minor);
        } else {
            printf("  Tensor Cores: not available (CC %d.%d)\n", prop.major, prop.minor);
        }
    }

    // Use device 0 for this demo
    CUDA_CHECK(cudaSetDevice(0));

    // Launch a 1×10 grid of threads → 10 total threads
    const dim3 gridDim(1);
    const dim3 blockDim(10);

    sanityCheckKernel<<<gridDim, blockDim>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return EXIT_SUCCESS;
}
