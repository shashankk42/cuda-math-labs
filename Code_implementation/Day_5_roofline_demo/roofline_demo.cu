/*
 * Day 4: roofline_demo.cu
 *
 * Compile:
 *   nvcc -O3 roofline_demo.cu -o roofline_demo
 *
 * Run:
 *   ./roofline_demo > roofline_data.csv
 *
 * Purpose:
 * - Launch & time a memory-bound kernel (vecAdd) and a compute-bound kernel (compHeavy)
 * - Calculate GFLOP/s, GB/s, and arithmetic intensity (FLOPs/Byte) for each
 * - Emit CSV for manual roofline plotting and later Nsight Roofline validation

 * ncu --set full --target-processes all --force-overwrite -o $HOME/ncu_reports/roofline_demo_full \
     /output/path/roofline_demo_wsl  
     
 * ncu-ui $HOME/ncu_reports/roofline_demo_full.ncu-rep     
*/


#include <cuda_runtime.h>
#include <iostream>

#define N (1<<24)            // total elements (~16 M)
#define ITER 1024            // work per element in compute kernel

// Memory‐bound: 1 add per element, 2 loads + 1 store → 3×4 B = 12 B, 1 flop
__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

// Compute‐bound: 1 load + 1 store + ITER FMAs each element
__global__ void compHeavy(float* D, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        float x = D[i];
        #pragma unroll
        for (int k = 0; k < ITER; ++k) {
            x = x * 1.0000001f + 0.0000001f;  // one FMA = 2 FLOPs, but count 1 here for simplicity
        }
        D[i] = x;
    }
}

static float elapsed_ms(cudaEvent_t s, cudaEvent_t e){
    float m=0; cudaEventElapsedTime(&m,s,e); return m;
}

int main(){
    int n = N;
    size_t bytes = n * sizeof(float);
    float *h = (float*)malloc(bytes);
    for(int i=0;i<n;++i) h[i] = i;

    float *dA, *dB, *dC, *dD;
    cudaMalloc(&dA,bytes); cudaMalloc(&dB,bytes);
    cudaMalloc(&dC,bytes); cudaMalloc(&dD,bytes);
    cudaMemcpy(dA,h,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,h,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dD,h,bytes,cudaMemcpyHostToDevice);

    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    int t=256, b=(n+t-1)/t;

    // Measure vecAdd
    cudaEventRecord(s);
    vecAdd<<<b,t>>>(dA,dB,dC,n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float msAdd = elapsed_ms(s,e);

    double flopsAdd = double(n) * 1.0;               // 1 add per element
    double bytesAdd = double(n) * 3 * sizeof(float); // 2 loads + 1 store
    double aiAdd   = flopsAdd / bytesAdd;
    double gflopAdd= flopsAdd/(msAdd*1e-3)/1e9;
    double bwAdd   = bytesAdd/(msAdd*1e-3)/1e9;

    // Measure compHeavy
    cudaEventRecord(s);
    compHeavy<<<b,t>>>(dD,n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float msComp = elapsed_ms(s,e);

    double flopsComp = double(n) * ITER;              // ITER “flops” per element
    double bytesComp = double(n) * 2 * sizeof(float); // 1 load + 1 store
    double aiComp   = flopsComp / bytesComp;
    double gflopComp= flopsComp/(msComp*1e-3)/1e9;
    double bwComp   = bytesComp/(msComp*1e-3)/1e9;

    // CSV output: Kernel,Time(ms),GFLOP/s,GB/s,AI(Flop/Byte)
    std::cout
      << "Kernel,Time_ms,GFLOP_s,GB_s,AI\n"
      << "vecAdd,"  << msAdd  << "," << gflopAdd << "," << bwAdd  << "," << aiAdd  << "\n"
      << "compHeavy,"<< msComp << "," << gflopComp<< "," << bwComp << "," << aiComp << "\n";

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
    free(h);
    return 0;
}
