# CUDA Math Labs

High-performance CUDA math library implementations and contributions for NVIDIA CUDA Math Libraries.

## Day 1: Sanity Check

Scripts Created/ Updated: - **sanity_check.cu**

### What I Did

- Wrote a 10-line `printf("Hello from block %d thread %d")` kernel.
- Compiled with `nvcc -arch=sm_89 -run`.
- Verified device properties (`cudaGetDeviceProperties`) match driver.

### Key Take Aways

- A kernel launch is `<<<grid, block>>>`; each thread sees its own block/thread IDs.
- Always check `cudaDeviceProp.major/minor` before enabling Tensor Cores.

### What I Read

- CUDA Programming Guide §5 “Programming Model”.

## Day 2: VectorAdd Kernels & Bandwidth Benchmark

Scripts Created/ Updated: - **vectorAdd_global_fixed.cu**, **vectorAdd_global_variable.cu**, **h2d_bandwidth_test.cu**, **alloc_benchmark.cu**

### What I Did

- Executed vector addition on fixed-size global-memory arrays.
- Extended the previous kernel to accept runtime-specified vector sizes.
- Measured Host‑to‑Device transfer bandwidth via cudaMemcpyAsync, comparing pageable memory (allocated with malloc) to pinned memory (allocated with cudaMallocHost).
- Timed and contrast allocation and transfer performance between pageable vs. pinned memory.

### Key Take Aways

- Transfers from pageable memory incur an internal staging copy, whereas pinned memory allows direct DMA transfers via PCIe, yielding much higher bandwidth. 
- Observed H2D bandwidth for pageable memory is typically around ~5-8 GB/s, while pinned memory approaches ~20-25 GB/s—up to ~3-5x faster. 
- Use of asynchronous memcpy with pinned buffers enables overlapping transfer and compute workloads. 
- Allocation of pinned memory has non‑trivial overhead and reduces available physical RAM—so it’s best used when buffers are reused across multiple transfers. 

### What I Read

- CUDA Programming Guide: Memory Management & Data Transfers sections.

- NVIDIA blogs & forums: discussions of pageable vs. pinned memory trade‑offs and benchmarking practices.

## Day 3: Shared‑Memory VectorAdd (Tiled Grid‑Stride)

Scripts Created/ Updated: - **vectorAdd_shared.cu**

### What I Did

- Switched to C++ style code 
- Implemented a grid‑stride, tiled vector add using dynamic shared memory (sA, sB).
- Applied stage -> __syncthreads() -> compute -> __syncthreads() with coalesced global loads/stores.
- Guarded partial tiles (i < N, zero‑fill), parsed digits‑only N (default 2^24), and validated via max |error|.

### Key Take Aways

- __syncthreads() is block‑wide; pad out of bound lanes.
- Dynamic shared memory gives launch‑time tile sizing and contiguous packing; tiling preserves coalescing.
- For pure vector add there’s no reuse, so shared memory is for learning the tiling + barrier discipline that does matter in reductions, stencils, and GEMM‑style kernels.

### What I Read

- CUDA C++ Programming Guide: Shared Memory; Thread Synchronization.
- CUDA Best Practices Guide: Memory Coalescing; Grid‑Stride Loops.


## Day 4: GEMM — Naive vs Tiled vs cuBLAS 

Scripts Created/ Updated: - **compare_with_cublas.cu**

### What I Did
- Implemented two FP32 GEMM kernels:
    - naive: each thread computes one C(i,j) with direct global loads.
    - tiled: blocks cooperatively load T×T tiles of A and B into dynamic shared memory and reuse them across the inner‐k loop.

- Accepted runtime sizes and options: ./compare_with_cublas M N K --kernel naive|tiled --tile T.
- Timed kernels with CUDA events; computed FLOPs as 2*M*N*K, then reported GFLOP/s.
- Called cuBLAS SGEMM on the same inputs and reported its GFLOP/s.


### Key Take Aways

- Shared-memory tiling increases data reuse and reduces global memory traffic.
- Use CUDA events for stable kernel timings (record, synchronize, elapsed_ms).
- cuBLAS SGEMM is a strong baseline. (8-9x faster compared to manual tiled GEMM implementation)
    - Need to read more for the reasons

### What I Read: 

- CUDA C++ Programming Guide: Matrix Multiply & Shared Memory tiling patterns.
- cuBLAS Library User Guide: SGEMM behavior, math modes, and Tensor Core notes.
- CUTLASS docs/tutorials: block-tiling, warp-tiling, pipelining, and epilogue design.
- Roofline model articles: relating Arithmetic Intensity to achievable performance.