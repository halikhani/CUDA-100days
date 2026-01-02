#include <stdio.h>
#include <cuda_runtime.h>

__global__ void prefixSumInclusive(const int* input, int* output, int n) {
    // shared memory
    extern __shared__ int sharedMem[];

    int tid = threadIdx.x;
    int globalIndex = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (globalIndex < n){
        sharedMem[tid] = input[globalIndex];
    } else {
        sharedMem[tid] = 0;
    }
    __syncthreads();

    // Parallel inclusive scan in shared memory
    // stride = 1, 2, 4, 8, 16, ...
    for (int stride = 1; stride < blockDim.x; stride *= 2){
        if (tid >= stride){
            sharedMem[tid] += sharedMem[tid - stride];
        }
        __syncthreads(); // make sure updates are visible before the next stride
    }

    // write back to global memory
    if (globalIndex < n) {
        output[globalIndex] = sharedMem[tid];
    }
}

int main(int argc, char** argv){
    const int N = 20;
    const int blockSize = 16;

    int h_input[N]  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    int h_output[N] = {0};

    int *d_input, *d_output;
    size_t size = N * sizeof(int);

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Launch 1 block, blockSize threads, and blockSize * sizeof(int) bytes of shared memory
    dim3 gridDim((N + blockSize - 1) / blockSize);
    dim3 blockDim(blockSize);
    size_t sharedMemSize = blockSize * sizeof(int);

    prefixSumInclusive<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Input:  ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\nOutput: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
    
}

        

