#include <iostream>
#include <stdio.h>

#define MASK_WIDTH 5
#define TILE_WIDTH 16
#define SHARED_SIZE (TILE_WIDTH + MASK_WIDTH - 1)

__constant__ float d_mask[MASK_WIDTH][MASK_WIDTH];

__global__ void conv2d_with_tiling(const float *A, float *C, int n){
    __shared__ float tile[SHARED_SIZE][SHARED_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // indices of the output location
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;

    //input coordinates of A for loading into shared memory
    int row_i = row_o - MASK_WIDTH / 2;
    int col_i = col_o - MASK_WIDTH / 2;

    // load data into shared memory with halo (row and col are assigned via ty and tx respectively)
    if (row_i >= 0 && row_i < n && col_i >= 0 && col_i < n)
        tile[ty][tx] = A[row_i * n + col_i];
    else
        tile[ty][tx] = 0.0f;
    
    __syncthreads();

    // perform convolution withing the output area
    if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < n && col_o < n){
        float result = 0.0f;
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                // use either flipped mask or not flipped mask
                // result += tile[ty + i][tx + j] * d_mask[i][j];
                result += tile[ty + i][tx + j] * d_mask[MASK_WIDTH - 1 - i][MASK_WIDTH - 1 - j];
            }
        }
        C[row_o * n + col_o] = result;
    }
    
}

void checkCudaError(const char *msg){
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        fprintf(stderr, "%s failed with error %s\n", msg, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main(){
    const int n = 32;
    float *h_A = (float *)malloc(n * n * sizeof(float));
    float *h_C = (float *)malloc(n * n * sizeof(float));

    float h_mask[MASK_WIDTH][MASK_WIDTH];

    for (int i = 0; i < MASK_WIDTH; ++i)
        for (int j = 0; j < MASK_WIDTH; ++j)
            h_mask[i][j] = 1.0f;  // Simple averaging kernel
    
    for (int i = 0; i < n * n; ++i)
        h_A[i] = 1.0f;

    float *d_A, *d_C;
    cudaMalloc((void **)&d_A, n * n * sizeof(float));
    cudaMalloc((void **)&d_C, n * n * sizeof(float));
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, h_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float));
    checkCudaError("Mask copy");

    dim3 blockDim(SHARED_SIZE, SHARED_SIZE);
    dim3 gridDim((n + TILE_WIDTH - 1)/TILE_WIDTH, (n + TILE_WIDTH - 1)/TILE_WIDTH);
    conv2d_with_tiling<<<gridDim, blockDim>>>(d_A, d_C, n);
    cudaDeviceSynchronize();
    checkCudaError("Kernel run");

    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("Output copy");

    // Debug output
    printf("Output matrix C:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.1f ", h_C[i * n + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);

    return 0;
}