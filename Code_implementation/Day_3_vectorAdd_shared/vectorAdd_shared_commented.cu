// Day 3 — Core shared-memory vector add (CUDA/C++)
// Purpose: demonstrate the essential "stage -> sync -> compute -> sync" pattern.
//
// Kernel processes tiles of size blockDim.x elements. For each tile:
//   1) Each thread loads one element from A and B (global -> shared).
//   2) __syncthreads() to make the staged data visible to the whole block.
//   3) Each thread writes C = A + B back to global.
//   4) __syncthreads() before reusing shared memory for the next tile.
//
// Note: For plain vector add there is no data reuse; shared memory is used here
// purely to practice the correct tiling + barrier pattern.

// Compile with:
//   nvcc -O2 -arch=sm_89 vectorAdd_shared_commented.cu -o vectorAdd_shared_commented
// Run:
//   ./vectorAdd_shared_commented
//   ./vectorAdd_shared_commented <N>   // optional: N = number of elements (digits only)

#include <cuda_runtime.h>  // CUDA runtime API
#include <iostream>        // std::cout, std::cerr

// -----------------------------
// Kernel: shared-memory–staged vector add
// -----------------------------
__global__ void vecAddSharedBasic(const float* A,   // input A (global memory)
                                  const float* B,   // input B (global memory)
                                  float* C,         // output C (global memory)
                                  size_t N)         // number of elements
{
    // Dynamic shared memory buffer provided at launch.
    // We split it into two tiles: sA and sB, each of length blockDim.x floats.
    extern __shared__ float tile[];
    float* sA = tile;                    // A tile in shared memory
    float* sB = tile + blockDim.x;       // B tile in shared memory (directly after sA)

    // Total elements covered by the whole grid each loop: gridDim.x * blockDim.x
    const size_t tileSpan = static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);

    // Each block starts at a different base index and grid-strides by tileSpan.
    for (size_t base = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x);
         base < N;
         base += tileSpan)
    {
        // Global index handled by this thread for the current tile.
        const size_t i = base + static_cast<size_t>(threadIdx.x);

        // (1) Stage global → shared (unit-stride across threads → coalesced).
        if (i < N) {
            sA[threadIdx.x] = A[i];
            sB[threadIdx.x] = B[i];
        } else {
            // Define shared contents for out-of-bounds lanes of the last tile.
            sA[threadIdx.x] = 0.0f;
            sB[threadIdx.x] = 0.0f;
        }

        // Ensure all threads finished staging before any reads from shared.
        __syncthreads();

        // (2) Compute from shared and write back to global (also coalesced).
        if (i < N) {
            C[i] = sA[threadIdx.x] + sB[threadIdx.x];
        }

        // Ensure no thread is still using this tile before reusing shared memory.
        __syncthreads();
    }
}

int main(int argc, char** argv)
{
    // -----------------------------
    // Parse N (digits only), default N = 2^24
    // -----------------------------
    size_t N = (1ull << 24);
    if (argc > 1) {
        size_t val = 0;
        const char* p = argv[1];
        if (*p == '\0') { std::cerr << "Invalid N\n"; return 1; }
        while (*p) {
            if (*p < '0' || *p > '9') { std::cerr << "Invalid N\n"; return 1; }
            size_t d = static_cast<size_t>(*p - '0');
            val = val * 10 + d;
            ++p;
        }
        N = val;
    }
    std::cout << "N = " << N << " elements\n";

    // -----------------------------
    // Host allocations (C++ new/delete)
    // -----------------------------
    float* hA = new float[N];
    float* hB = new float[N];
    float* hC = new float[N];

    // Initialize inputs with a simple pattern.
    for (size_t i = 0; i < N; ++i) {
        hA[i] = static_cast<float>(i);          // A[i] = i
        hB[i] = static_cast<float>(2.0f * i);   // B[i] = 2*i
    }

    // -----------------------------
    // Device allocations
    // -----------------------------
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    if (cudaMalloc(&dA, N * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dB, N * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dC, N * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "Device allocation failed: " << cudaGetErrorString(cudaGetLastError()) << "\n";
        delete[] hA; delete[] hB; delete[] hC;
        return 1;
    }

    // Copy inputs to device.
    if (cudaMemcpy(dA, hA, N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dB, hB, N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "H2D memcpy failed: " << cudaGetErrorString(cudaGetLastError()) << "\n";
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        delete[] hA; delete[] hB; delete[] hC;
        return 1;
    }

    // -----------------------------
    // Launch configuration
    // -----------------------------
    const int block = 128;                                      // tile size per block
    const int grid  = static_cast<int>((N + block - 1) / block); // enough blocks to cover N
    const size_t shmemBytes = static_cast<size_t>(2 * block * sizeof(float)); // sA + sB

    // Launch kernel with dynamic shared memory.
    vecAddSharedBasic<<<grid, block, shmemBytes>>>(dA, dB, dC, N);

    // Check for launch/runtime errors and wait for completion.
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaGetLastError()) << "\n";
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        delete[] hA; delete[] hB; delete[] hC;
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(cudaGetLastError()) << "\n";
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        delete[] hA; delete[] hB; delete[] hC;
        return 1;
    }

    // -----------------------------
    // Copy back and validate (max |error| without extra libs)
    // -----------------------------
    if (cudaMemcpy(hC, dC, N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "D2H memcpy failed: " << cudaGetErrorString(cudaGetLastError()) << "\n";
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        delete[] hA; delete[] hB; delete[] hC;
        return 1;
    }

    double maxAbsErr = 0.0;
    for (size_t i = 0; i < N; i++) {
        // Reference sum is simple with our initialization: ref = A[i] + B[i]
        float ref_f = hA[i] + hB[i];
        double diff = (double)hC[i] - (double)ref_f;
        if (diff < 0.0) diff = -diff;           // manual absolute value
        if (diff > maxAbsErr) maxAbsErr = diff;
    }
    std::cout << "max |error| = " << maxAbsErr << "\n";

    // -----------------------------
    // Cleanup
    // -----------------------------
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
    return 0;
}
