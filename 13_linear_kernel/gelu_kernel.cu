#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void gelu_kernel(float* data, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        data[idx] = 0.5 * (1.0 + erf(data[idx]/sqrt(2.0)));
    }
}

int main(){
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    printf("Number of devices: %d\n", dev_count);

    cudaDeviceProp dev_prop;
    for (int i = 0; i < dev_count; i++){
        cudaGetDeviceProperties(&dev_prop, i);
        printf("Device %d: %s\n", i, dev_prop.name);
        printf("Total global memory: %zu bytes\n", dev_prop.totalGlobalMem);
        printf("Shared memory per block: %zu bytes\n", dev_prop.sharedMemPerBlock);
        printf("Registers per block: %d\n", dev_prop.regsPerBlock);
        printf("Warp size: %d\n", dev_prop.warpSize);
        printf("Maximum threads per block: %d\n", dev_prop.maxThreadsPerBlock);
        printf("Maximum threads per multiprocessor: %d\n", dev_prop.maxThreadsPerMultiProcessor);
        printf("Number of SMs: %d\n", dev_prop.multiProcessorCount);
        printf("Clock rate: %d MHz\n", dev_prop.clockRate);
    }

    const int N = 1024;
    float A[N];
    for (int i = 0; i < N; i++){
        A[i] = -1*(float)i/2;
    }

    for (int i = 0; i < 10; ++i) {
        std::cout << "A[" << i << "]: " << A[i] << std::endl;
    }

    float *d_A;
    cudaMalloc((void**)&d_A, N*sizeof(float));
    cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    gelu_kernel<<<gridSize, blockSize>>>(d_A, N);
    cudaDeviceSynchronize();
    cudaMemcpy(A, d_A, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);

    for (int i = 0; i < 10; ++i) {
        std::cout << "A[" << i << "]: " << A[i] << std::endl;
    }

    return 0;
}