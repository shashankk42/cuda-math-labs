/***************************************************************
* compare_with_cublas.cu  (CUDA/C++)
*
* Compare your GEMM (naive or tiled) with cuBLAS SGEMM.
*
* Build:
*   nvcc -O3 -arch=sm_89 compare_with_cublas.cu -o compare_with_cublas -lcublas
*
* Run:
*   ./compare_with_cublas M N K --kernel naive|tiled --tile 32
***************************************************************/
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <cstdlib>

#ifndef TILE
#define TILE 32
#endif

// --------------------------- Your kernels ----------------------------
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col
    if (i >= M || j >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = acc;
}

__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K, int T) {
    extern __shared__ float smem[];
    float* As = smem;              // T*T
    float* Bs = smem + T*T;        // T*T

    int ty = threadIdx.y, tx = threadIdx.x;
    int i = blockIdx.y * T + ty;
    int j = blockIdx.x * T + tx;

    float acc = 0.f;
    int tiles = (K + T - 1) / T;

    for (int t = 0; t < tiles; ++t) {
        int a_col = t * T + tx;    // col in A
        int b_row = t * T + ty;    // row in B

        // Load tiles (guarded)
        As[ty * T + tx] = (i < M && a_col < K) ? A[i * K + a_col] : 0.f;
        Bs[ty * T + tx] = (b_row < K && j < N) ? B[b_row * N + j] : 0.f;

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) { // NOTE: assumes T == TILE at compile-time, see build note
            acc += As[ty * T + k] * Bs[k * T + tx];
        }
        __syncthreads();
    }
    if (i < M && j < N) C[i * N + j] = acc;
}

// --------------------------- Timing helper ---------------------------
static inline float elapsed_ms(cudaEvent_t s, cudaEvent_t e) {
    float ms = 0.f; cudaEventElapsedTime(&ms, s, e); return ms;
}

// --------------------------- Main ------------------------------------
int main(int argc, char** argv) {
    // Defaults
    int M = 2048, N = 2048, K = 2048;
    std::string kernel = "tiled";
    int T = TILE;

    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    for (int i = 4; i < argc; ++i) {
        if (std::string(argv[i]) == "--kernel" && i + 1 < argc) {
            kernel = argv[++i];
        } else if (std::string(argv[i]) == "--tile" && i + 1 < argc) {
            T = std::atoi(argv[++i]);
        }
    }

    if (kernel == "tiled" && T != TILE) {
        std::cout << "[Note] Rebuild with -DTILE=" << T
                  << " to unroll the inner loop exactly for your chosen tile.\n";
    }

    const size_t bytesA = size_t(M) * K * sizeof(float);
    const size_t bytesB = size_t(K) * N * sizeof(float);
    const size_t bytesC = size_t(M) * N * sizeof(float);

    // Host init
    std::vector<float> hA(size_t(M) * K);
    std::vector<float> hB(size_t(K) * N);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = float((i % 7) - 3) * 0.25f;
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = float((i % 5) - 2) * 0.5f;

    // Device buffers
    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC);
    cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, bytesC);

    // FLOPs for GEMM
    const double flops = 2.0 * double(M) * N * K;

    // Events
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);

    // ---------------- Your kernel timing ----------------
    float ms_yours = 0.f;
    if (kernel == "naive") {
        dim3 b(32, 32), g((N + b.x - 1) / b.x, (M + b.y - 1) / b.y);
        // warmup
        for (int w = 0; w < 2; ++w) gemm_naive<<<g, b>>>(dA, dB, dC, M, N, K);
        cudaEventRecord(s);
        gemm_naive<<<g, b>>>(dA, dB, dC, M, N, K);
        cudaEventRecord(e); cudaEventSynchronize(e);
        ms_yours = elapsed_ms(s, e);
    } else { // tiled (default)
        dim3 b(T, T), g((N + T - 1) / T, (M + T - 1) / T);
        size_t shmem = 2 * size_t(T) * T * sizeof(float); // As + Bs
        // warmup
        for (int w = 0; w < 2; ++w) gemm_tiled<<<g, b, shmem>>>(dA, dB, dC, M, N, K, T);
        cudaEventRecord(s);
        gemm_tiled<<<g, b, shmem>>>(dA, dB, dC, M, N, K, T);
        cudaEventRecord(e); cudaEventSynchronize(e);
        ms_yours = elapsed_ms(s, e);
    }
    const double gflops_yours = (flops / 1e9) / (ms_yours * 1e-3);

    std::cout << "[Yours: " << kernel << (kernel=="tiled" ? (" (T=" + std::to_string(T) + ")") : "") << "]\n"
              << "  Time: " << std::fixed << std::setprecision(3) << ms_yours << " ms\n"
              << "  GFLOP/s: " << std::setprecision(2) << gflops_yours << "\n";

    // ---------------- cuBLAS SGEMM timing ----------------
    // Row-major A(MxK), B(KxN), C(MxN):
    // Use trick: C^T = B^T * A^T with column-major cuBLAS
    cublasHandle_t h; cublasCreate(&h);
    const float alpha = 1.0f, beta = 0.0f;

    // warmup
    for (int w = 0; w < 2; ++w) {
        cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T,
                    N, M, K,
                    &alpha,
                    dB, N,   // B^T: N x K, ldb=N (since B is KxN row-major)
                    dA, K,   // A^T: K x M, lda=K
                    &beta,
                    dC, N);  // C^T: N x M, ldc=N (stores into row-major C)
    }
    cudaEventRecord(s);
    cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T,
                N, M, K,
                &alpha, dB, N, dA, K, &beta, dC, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    const float ms_cublas = elapsed_ms(s, e);
    const double gflops_cublas = (flops / 1e9) / (ms_cublas * 1e-3);

    std::cout << "[cuBLAS SGEMM]\n"
              << "  Time: " << std::fixed << std::setprecision(3) << ms_cublas << " ms\n"
              << "  GFLOP/s: " << std::setprecision(2) << gflops_cublas << "\n";

    // ---------------- Comparison ----------------
    std::cout << "[Comparison]\n"
              << "  Speedup (cuBLAS / yours): "
              << std::setprecision(3) << (gflops_cublas / gflops_yours) << "×\n";

    // Cleanup
    cublasDestroy(h);
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
