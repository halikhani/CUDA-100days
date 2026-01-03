#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA kernel for parallel reduction (sum)
// This kernel performs a tree-based reduction where each block reduces its portion of data
__global__ void reductionKernel(float* X, float* output, int n) {
    // Shared memory for each block - stores partial sums
    // SIZE should be equal to blockDim.x (number of threads per block)
    extern __shared__ float partial_sum[];
    
    // Step 1: Each thread loads one element from global memory to shared memory
    // Global index: blockIdx.x * blockDim.x + threadIdx.x
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        partial_sum[threadIdx.x] = X[idx];
    } else {
        partial_sum[threadIdx.x] = 0.0f; // Pad with zeros if out of bounds
    }
    
    // Synchronize all threads in the block to ensure all data is loaded
    __syncthreads();
    
    // Step 2: Tree-based reduction
    // We perform a binary tree reduction where we halve the active threads each iteration
    unsigned int t = threadIdx.x;
    
    // Start with stride = blockDim.x / 2, then halve it each iteration
    // Example: If blockDim.x = 8, stride goes: 4 -> 2 -> 1
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        // Only threads with index < stride participate in this iteration
        // Each active thread adds the value from its "partner" (stride positions away)
        if (t < stride) {
            partial_sum[t] += partial_sum[t + stride];
        }
        // Synchronize before next iteration to ensure all additions complete
        __syncthreads();
    }
    
    // Step 3: Thread 0 writes the final reduced sum for this block to global memory
    if (threadIdx.x == 0) {
        output[blockIdx.x] = partial_sum[0];
    }
}

// Helper function to check CUDA errors
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(1);
    }
}

int main() {
    // Test parameters
    const int N = 1024;  // Total number of elements
    const int BLOCK_SIZE = 256;  // Threads per block
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;  // Number of blocks needed
    
    printf("=== CUDA Reduction Test ===\n");
    printf("Total elements: %d\n", N);
    printf("Threads per block: %d\n", BLOCK_SIZE);
    printf("Number of blocks: %d\n", NUM_BLOCKS);
    
    // Allocate host memory
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(NUM_BLOCKS * sizeof(float));
    
    // Initialize input array with test data
    printf("\nInitializing input array...\n");
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f; 
    }
    
    // Calculate expected sum (for verification)
    float expected_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        expected_sum += h_input[i];
    }
    printf("Expected sum: %.2f\n", expected_sum);
    
    // Allocate device memory
    float* d_input;
    float* d_output;
    checkCudaError(cudaMalloc((void**)&d_input, N * sizeof(float)), "Failed to allocate d_input");
    checkCudaError(cudaMalloc((void**)&d_output, NUM_BLOCKS * sizeof(float)), "Failed to allocate d_output");
    
    // Copy input data to device
    checkCudaError(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy input to device");
    
    // Launch kernel
    printf("\nLaunching reduction kernel...\n");
    reductionKernel<<<NUM_BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_input, d_output, N);
    
    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    // Wait for kernel to complete
    checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize");
    
    // Copy results back to host
    checkCudaError(cudaMemcpy(h_output, d_output, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy output from device");
    
    // Final reduction on CPU (sum all block results)
    float final_sum = 0.0f;
    printf("\nBlock results:\n");
    for (int i = 0; i < NUM_BLOCKS; i++) {
        printf("  Block %d: %.2f\n", i, h_output[i]);
        final_sum += h_output[i];
    }
    
    // Verify result
    printf("\n=== Results ===\n");
    printf("Computed sum: %.2f\n", final_sum);
    printf("Expected sum: %.2f\n", expected_sum);
    printf("Difference: %.6f\n", fabs(final_sum - expected_sum));
    
    if (fabs(final_sum - expected_sum) < 0.01f) {
        printf("✓ Test PASSED! Results match.\n");
    } else {
        printf("✗ Test FAILED! Results don't match.\n");
    }
    
    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
