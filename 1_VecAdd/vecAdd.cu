#include <iostream>
#include <cuda_runtime.h>

__global__ void vecAdd(const float* A, const float* B, float* C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        C[i] = A[i] + B[i];
    }
}

int main(){
    const int N = 10;
    float A[N], B[N], C[N];
    float *d_a, *d_b, *d_c;

    // Initialize inputs
    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));

    cudaMemcpy(d_a, A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N*sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    // int gridsize = ceil(N/blocksize);
    int gridSize = (N + blockSize - 1) / blockSize;

    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    // Wait for GPU to finish
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "A + B = C:\n";
    for (int i = 0; i < N; i++) {
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}