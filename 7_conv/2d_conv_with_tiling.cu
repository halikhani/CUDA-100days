#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>


#define MASK_WIDTH 5
#define TILE_WIDTH 16
#define SHARED_SIZE (TILE_WIDTH + MASK_WIDTH - 1)
__constant__ float Mask[MASK_WIDTH][MASK_WIDTH];


__global__ void conv_2d_tiled_kernel(const float *A, float *C, int n){
    __shared__ float tile[SHARED_SIZE][SHARED_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Base coordinates for the input tile (including halo)
    int base_row = blockIdx.y * TILE_WIDTH - MASK_WIDTH / 2;
    int base_col = blockIdx.x * TILE_WIDTH - MASK_WIDTH / 2;

    // Input coordinates for loading into shared memory
    int row_i = base_row + ty;
    int col_i = base_col + tx;

    // Load data into shared memory (all SHARED_SIZE x SHARED_SIZE threads participate)
    // if the input coordinates are within the bounds of the input matrix, load the data into the shared memory
    // otherwise, its a halo cell so set it to 0
    if (row_i >= 0 && row_i < n && col_i >= 0 && col_i < n)
        tile[ty][tx] = A[row_i * n + col_i];
    else
        tile[ty][tx] = 0.0f;

    // final note: Thread (ty, tx) loads input element at (base_row + ty, base_col + tx)
    __syncthreads();

    // Perform convolution (only TILE_WIDTH x TILE_WIDTH threads compute output since the output is only in the tile area)
    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
        // Output location indices
        int row_o = blockIdx.y * TILE_WIDTH + ty;
        int col_o = blockIdx.x * TILE_WIDTH + tx;

        float output = 0.0f;
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                output += Mask[i][j] * tile[ty + i][tx + j];
            }
        }
        if (row_o < n && col_o < n) {
            C[row_o * n + col_o] = output;
        }
    }
}

// CPU reference implementation for verification
void conv2d_cpu_reference(const float *input, float *output, const float mask[MASK_WIDTH][MASK_WIDTH], int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int i = 0; i < MASK_WIDTH; i++) {
                for (int j = 0; j < MASK_WIDTH; j++) {
                    int input_row = row + i - MASK_WIDTH / 2;
                    int input_col = col + j - MASK_WIDTH / 2;
                    if (input_row >= 0 && input_row < n && input_col >= 0 && input_col < n) {
                        sum += input[input_row * n + input_col] * mask[i][j];
                    }
                }
            }
            output[row * n + col] = sum;
        }
    }
}

// defining a function-like macro that takes one arg 'call'
// backlashes are used to continue the macro definition on the next line
// __FILE__: Expands to the current source filename (string literal).
// __LINE__: Expands to the current line number (integer).
// Since the condition is 0 (false), the loop runs exactly once, then stops.
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

int main() {
    const int n = 32;
    size_t size = n * n * sizeof(float);
    
    float *h_input = (float*)malloc(size);
    float *h_output_gpu = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);
    float h_mask[MASK_WIDTH][MASK_WIDTH];
    
    // Initialize data
    for (int i = 0; i < n * n; i++) h_input[i] = (float)(i % 10);
    for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++) 
        ((float*)h_mask)[i] = 1.0f / (MASK_WIDTH * MASK_WIDTH);
    
    conv2d_cpu_reference(h_input, h_output_cpu, h_mask, n);
    
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA(cudaMalloc((void**)&d_output, size));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(Mask, h_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float)));
    
    dim3 blockDim(SHARED_SIZE, SHARED_SIZE);
    dim3 gridDim((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH);
    conv_2d_tiled_kernel<<<gridDim, blockDim>>>(d_input, d_output, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost));
    
    // Verify
    bool correct = true;
    for (int i = 0; i < n * n; i++) {
        // fabsf is a floating point absolute value function
        if (fabsf(h_output_gpu[i] - h_output_cpu[i]) > 1e-4f) {
            std::cout << "Mismatch at index " << i << std::endl;
            correct = false;
            break;
        }
    }
    std::cout << (correct ? "✓ All results match!\n" : "✗ Results do not match!\n");
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    // return 0 if the results match, 1 otherwise
    return correct ? 0 : 1;
}