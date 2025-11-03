#include <iostream>
#include <stdio.h>

#define MASK_WIDTH 5
#define TILE_WIDTH 32
#define SHARED_SIZE (TILE_WIDTH + MASK_WIDTH - 1)

__constant__ float d_mask[MASK_WIDTH][MASK_WIDTH];

__global__ void conv2d_with_tiling(const float *A, float *C, int n){
    // TODO
}

void checkCudaError(const char *msg){
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        fprintf(stderr, "%s failed with error %s\n", msg, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main(){
    // TODO
}