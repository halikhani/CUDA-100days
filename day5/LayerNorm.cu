#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <math.h>

__global__ void layerNormSimple(const float* A, float* B, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute mean of the row
    float mean = 0.0f;

    for (int col = 0; col < cols; col++) {
        mean += A[row * cols + col];
    }
    mean /= cols;

    // Compute variance of the row
    float variance = 0.0f;
    for (int col = 0; col < cols; col++) {
        float diff = A[row*cols + col] - mean;
        variance += diff * diff;
    }
    variance /= cols;
    float inv_std = rsqrtf(variance + 1e-7f);

    // Normalize and write to output
    for (int col = 0; col < cols; col++) {
        B[row*cols + col] = (A[row*cols + col] - mean) * inv_std;
    }
}

__global__ void layerNormParallel(const float* __restrict__ A, float* __restrict__ B, int rows, int cols) {
    // one block per row
    int row = blockIdx.x; // which row this block is processing
    int tid = threadIdx.x; // thread within row

    if (row >= rows) return;

    // shared memory to store per row mean and variance
    __shared__ float mean_shared, invstd_shared;

    // compute mean and variance (thread 0 does it) ----
    if (tid == 0) {
        float sum = 0.0f;
        float sumsq = 0.0f;
    // Walk over all columns in this row (serial on thread 0)
    for (int c = 0; c < cols; c++) {
        float v = A[row * cols + c];
        sum   += v;
        sumsq += v * v;
    }
    float mean = sum / cols;
    float var = sumsq / cols - mean * mean;
    float inv_std = rsqrtf(var + 1e-7f);

    mean_shared   = mean;
    invstd_shared = inv_std;
    }
    __syncthreads();

    float mean   = mean_shared;
    float invstd = invstd_shared;

    // normalize in parallel
    // Each thread handles multiple columns with a stride of blockDim.x
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = A[row * cols + c];
        float norm = (v - mean) * invstd;
        B[row * cols + c] = norm;
    }

}

int main(int argc, char** argv) {
    const int rows = 10, cols = 10;
    float *A, *B;

    A = (float*)malloc(rows*cols*sizeof(float));
    B = (float*)malloc(rows*cols*sizeof(float));

    // Init
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A[i*cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    float *d_a, *d_b;
    cudaMalloc((void**)&d_a, rows*cols*sizeof(float));
    cudaMalloc((void**)&d_b, rows*cols*sizeof(float));

    cudaMemcpy(d_a, A, rows*cols*sizeof(float), cudaMemcpyHostToDevice);

    // int blockSize = 16;
    // int gridSize = (rows + blockSize - 1) / blockSize;
    int grid_size = rows;
    int block_size = 128;
    // layerNormSimple<<<gridSize, blockSize>>>(d_a, d_b, rows, cols);
    layerNormParallel<<<grid_size, block_size>>>(d_a, d_b, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(B, d_b, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

    // Print result and assert if layernorm is correct by summing up the row vals of b and print
    float sum = 0.0f;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += B[i*cols + j];
        }
        std::cout << "Row " << i << " sum: " << sum << std::endl;
        sum = 0.0f;
    }

    // Free memory
    free(A);
    free(B);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}

