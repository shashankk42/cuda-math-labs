# CUDA Math Labs

High-performance CUDA math library implementations and contributions for NVIDIA CUDA Math Libraries.

## Day 1: Sanity Check

### What I Did

- Wrote a 10-line `printf("Hello from block %d thread %d")` kernel.
- Compiled with `nvcc -arch=sm_89 -run`.
- Verified device properties (`cudaGetDeviceProperties`) match driver.

### Key Take Aways

- A kernel launch is `<<<grid, block>>>`; each thread sees its own block/thread IDs.
- Always check `cudaDeviceProp.major/minor` before enabling Tensor Cores.

### What I Read

- CUDA Programming Guide §5 “Programming Model”.
```
