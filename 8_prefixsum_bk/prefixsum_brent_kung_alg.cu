#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_SIZE (2 * BLOCK_SIZE)

__constant__ float dummy_mask[1];

__global__ void prefixsum_brent_kung(float* input, float* output, int n) {
    __shared__ float temp[TILE_SIZE];
    
    // each thread handles 2 elements in a block-wise tiled layout
    int tx = threadIdx.x;
    int start = 2 * blockIdx.x * blockDim.x;
    int i = start + tx;

    // Load input to shared memory
    temp[tx] = (i < n) ? input[i] : 0.0f;
    temp[tx + BLOCK_SIZE] = (i + BLOCK_SIZE < n) ? input[i + BLOCK_SIZE] : 0.0f;

    __syncthreads();

    // Up-sweep phase
    for (int stride = 1; stride < TILE_SIZE; stride *= 2) {
        int idx = (tx + 1) * stride * 2 - 1;
        if (idx < TILE_SIZE) {
            temp[idx] += temp[idx - stride];
        }
        __syncthreads();
    }

    // Down-sweep phase
    for (int stride = TILE_SIZE / 4; stride >= 1; stride /= 2) {
        int idx = (tx + 1) * stride * 2 - 1;
        if (idx + stride < TILE_SIZE)
            temp[idx + stride] += temp[idx];
        __syncthreads();
    }
    
    // Write to output
    if (i < n) output[i] = temp[tx];
    if (i + blockDim.x < n) output[i + blockDim.x] = temp[tx + blockDim.x];

}

int main() {
    const int N = 10;
    float h_input[N], h_output[N];
    for (int i = 0; i < N; ++i) h_input[i] = i + 1;

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE));

    prefixsum_brent_kung<<<grid, block>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Prefix Sum:\n");
    for (int i = 0; i < N; ++i) printf("%.1f ", h_output[i]);
    printf("\n");

    cudaFree(d_input); cudaFree(d_output);
    return 0;
}


