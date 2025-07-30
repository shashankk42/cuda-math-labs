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
