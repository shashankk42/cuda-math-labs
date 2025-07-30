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
//   nvcc -O2 -arch=sm_89 vectorAdd_shared.cu -o vectorAdd_shared
// Run:
//   ./vectorAdd_shared
//   ./vectorAdd_shared <N>   // optional: N = number of elements (digits only)

#include <cuda_runtime.h>  // CUDA runtime API
#include <iostream>        // std::cout, std::cerr

// -----------------------------
// Kernel: C = A + B using shared-memory tiles
// -----------------------------
__global__ void vecAddSharedBasic(const float* A,   // input A (global)
                                  const float* B,   // input B (global)
                                  float* C,         // output C (global)
                                  size_t N)         // element count
{
    // Dynamic shared memory: two contiguous tiles, each blockDim.x floats.
    extern __shared__ float tile[];
    float* sA = tile;                    // tile for A
    float* sB = tile + blockDim.x;       // tile for B (after sA)

    // Elements advanced per outer-loop iteration across the entire grid.
    const size_t tileSpan = static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);

    // Grid-stride loop over tiles; each block handles its lane each pass.
    for (size_t base = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x);
         base < N;
         base += tileSpan)
    {
        // Element index for this thread in the current tile.
        const size_t i = base + static_cast<size_t>(threadIdx.x);

        // (1) Stage: coalesced global → shared (pad OOB lanes for final partial tile).
        if (i < N) {
            sA[threadIdx.x] = A[i];
            sB[threadIdx.x] = B[i];
        } else {
            sA[threadIdx.x] = 0.0f;
            sB[threadIdx.x] = 0.0f;
        }

        // Barrier: ensure all threads finished staging before any shared reads.
        __syncthreads();

        // (2) Compute + store: read from shared, write C to global (coalesced).
        if (i < N) {
            C[i] = sA[threadIdx.x] + sB[threadIdx.x];
        }

        // Barrier: ensure no thread still uses this tile before reuse.
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
    // Host allocations
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

    // H2D copies.
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
    const int block = 128;                                       // threads/tile per block
    const int grid  = static_cast<int>((N + block - 1) / block); // blocks to cover N
    const size_t shmemBytes = static_cast<size_t>(2 * block * sizeof(float)); // sA + sB

    // Launch kernel with dynamic shared memory.
    vecAddSharedBasic<<<grid, block, shmemBytes>>>(dA, dB, dC, N);

    // Check launch and execution status.
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
    // Copy back and validate (max |error|)
    // -----------------------------
    if (cudaMemcpy(hC, dC, N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "D2H memcpy failed: " << cudaGetErrorString(cudaGetLastError()) << "\n";
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        delete[] hA; delete[] hB; delete[] hC;
        return 1;
    }

    double maxAbsErr = 0.0;
    for (size_t i = 0; i < N; i++) {
        // Reference: A[i] + B[i]
        float ref_f = hA[i] + hB[i];
        double diff = (double)hC[i] - (double)ref_f;
        if (diff < 0.0) diff = -diff;
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