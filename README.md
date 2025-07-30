# CUDA Math Labs

High-performance CUDA math library implementations and contributions for NVIDIA CUDA Math Libraries.

## Day 1: Sanity Check

### What I Did

- File: **sanity_check.cu**
- Wrote a 10-line `printf("Hello from block %d thread %d")` kernel.
- Compiled with `nvcc -arch=sm_89 -run`.
- Verified device properties (`cudaGetDeviceProperties`) match driver.

### Key Take Aways

- A kernel launch is `<<<grid, block>>>`; each thread sees its own block/thread IDs.
- Always check `cudaDeviceProp.major/minor` before enabling Tensor Cores.

### What I Read

- CUDA Programming Guide §5 “Programming Model”.

## Day 2: VectorAdd Kernels & Bandwidth Benchmark

### What I Did

- **vectorAdd_global_fixed.cu** – Executes vector addition on fixed-size global-memory arrays.

- **vectorAdd_global_variable.cu** – Extends the previous kernel to accept runtime-specified vector sizes.

- **h2d_bandwidth_test.cu** – Measures Host‑to‑Device transfer bandwidth via cudaMemcpyAsync, comparing pageable memory (allocated with malloc) to pinned memory (allocated with cudaMallocHost).

- **alloc_benchmark.cu** – Times and contrasts allocation and transfer performance between pageable vs. pinned memory.

### Key Take Aways

- Transfers from pageable memory incur an internal staging copy, whereas pinned memory allows direct DMA transfers via PCIe, yielding much higher bandwidth. 

- Observed H2D bandwidth for pageable memory is typically around ~5-8 GB/s, while pinned memory approaches ~20-25 GB/s—up to ~3-5x faster. 

- Use of asynchronous memcpy with pinned buffers enables overlapping transfer and compute workloads. 

- Allocation of pinned memory has non‑trivial overhead and reduces available physical RAM—so it’s best used when buffers are reused across multiple transfers. 

### What I Read

CUDA Programming Guide: Memory Management & Data Transfers sections.

NVIDIA blogs & forums: discussions of pageable vs. pinned memory trade‑offs and benchmarking practices.
```
