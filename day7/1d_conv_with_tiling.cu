#include <cuda_runtime.h>
#include <iostream>

#define N 10
#define MASK_WIDTH 5
#define h MASK_WIDTH/2

__constant__ float d_mask[MASK_WIDTH];

__global__ void conv1d_tiled(float *A, float *C, int n){
    int t = threadIdx.x;    // thread idx in block
    int B = blockDim.x;     // block size
    int i = blockIdx.x * B + t;    // global thread idx

    extern __shared__ float sA[];

    // load center B elements of A into shared memory
    if (i < n) sA[t + h] = A[i];
    else sA[t + h] = 0.0f;

    // load left halo
    if (t < h){
        int left_idx = i - h;
        sA[t] = left_idx < 0 ? 0.0f : A[left_idx];
    }

    // load right halo
    if (t < h){
        int right_idx = i + B;
        sA[t + B + h] = right_idx >= n ? 0.0f : A[right_idx];
    }

    __syncthreads();

    // conv with flipped mask
    if (i < n){
        float sum = 0.0f;
        for (int k = 0; k < MASK_WIDTH; k++){
            sum += sA[t + k] * d_mask[MASK_WIDTH - 1 - k];
        }
        C[i] = sum;
    }
}

int main(){
    float h_A[N], h_C[N], h_M[MASK_WIDTH];
    
    // init input
    for (int i = 0; i < N; i++){
        h_A[i] = i;
    }

    // init mask [0, 1, 2, 3, 4]
    for (int i = 0; i < MASK_WIDTH; i++) h_M[i] = (float)i;

    float *d_A, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, h_M, MASK_WIDTH*sizeof(float));

    int blockSize = 8;
    int gridSize = (N + blockSize - 1) / blockSize;
    size_t shared = (blockSize + MASK_WIDTH - 1) * sizeof(float);

    conv1d_tiled<<<gridSize, blockSize, shared>>>(d_A, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // ---- Print results ----
    printf("Input:  ");
    for (int i = 0; i < N; i++) printf("%.0f ", h_A[i]);
    printf("\nMask:   ");
    for (int i = 0; i < MASK_WIDTH; i++) printf("%.0f ", h_M[i]);
    printf("\nOutput: ");
    for (int i = 0; i < N; i++) printf("%.0f ", h_C[i]);
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_C);
    return 0;
    
    
}


