#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

// Define TILE_WIDTH if not already defined
#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif

__global__ void conv1d_tiled_caching_kernel(float *N, float *P, const float *M, int Mask_Width, int Width){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // global thread idx
    __shared__ float N_ds[TILE_WIDTH];

    // Only process valid indices
    if (i >= Width) return;

    // load center elements (without halo) into shared memory
    N_ds[threadIdx.x] = N[i]; // thread idx is the local thread location and i is the global one 
    __syncthreads();

    int This_tile_start_point = blockIdx.x * blockDim.x;
    int This_tile_end_point = (blockIdx.x + 1) * blockDim.x;
    int N_start_point = i - Mask_Width / 2; // considering halo cells
    float PValue = 0.0f;
    
    // performing convolution for each tile
    for (int j = 0; j < Mask_Width; j++){
        int N_index = N_start_point + j;
        // next if checks if the current access to the N element falls within the input array
        // if not, means that its a halo cell and we just skip it
        if (N_index >= 0 && N_index < Width){
            // test if the current access to the N element falls within tile by testing against 
            // This_tile_start_point and This_tile_end_point
            // if yes, its an internal elements and can be accessed by N_ds
            // if not, its accessed from the N array directly, which is hopefully in the L2 cache
            if ((N_index >= This_tile_start_point) && (N_index < This_tile_end_point)){
                // FIX: Calculate local index directly from N_index instead of using 
                // threadIdx.x + j - Mask_Width / 2, which is error-prone
                int local_idx = N_index - This_tile_start_point;
                // Additional safety check (though mathematically should not be needed)
                if (local_idx >= 0 && local_idx < TILE_WIDTH) {
                    PValue += N_ds[local_idx] * M[j];
                }
            }
            else{
                PValue += N[N_index] * M[j];
            }
        }
    }
    P[i] = PValue;
}

// CPU reference implementation for verification
void conv1d_cpu_reference(float *input, float *output, const float *mask, int mask_width, int width){
    for (int i = 0; i < width; i++){
        float sum = 0.0f;
        for (int j = 0; j < mask_width; j++){
            int input_idx = i + j - mask_width / 2;
            if (input_idx >= 0 && input_idx < width){
                sum += input[input_idx] * mask[j];
            }
        }
        output[i] = sum;
    }
}

// Helper function to check CUDA errors
void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << message << " - CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(){
    // Test parameters
    const int Width = 16;
    const int Mask_Width = 5;
    const int blockSize = TILE_WIDTH;  // Use TILE_WIDTH as block size
    
    // Host arrays
    float h_input[Width];
    float h_output[Width];
    float h_output_cpu[Width];  // For CPU reference
    float h_mask[Mask_Width];
    
    // Initialize input array with test data
    for (int i = 0; i < Width; i++){
        h_input[i] = (float)i;  // Simple test: [0, 1, 2, ..., Width-1]
    }
    
    // Initialize mask (simple averaging-like kernel: [0, 1, 2, 1, 0])
    h_mask[0] = 0.0f;
    h_mask[1] = 1.0f;
    h_mask[2] = 2.0f;
    h_mask[3] = 1.0f;
    h_mask[4] = 0.0f;
    
    // Compute CPU reference
    conv1d_cpu_reference(h_input, h_output_cpu, h_mask, Mask_Width, Width);
    
    // Device arrays
    float *d_input, *d_output, *d_mask;
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, Width * sizeof(float));
    checkCudaError("Failed to allocate device memory for input");
    
    cudaMalloc((void**)&d_output, Width * sizeof(float));
    checkCudaError("Failed to allocate device memory for output");
    
    cudaMalloc((void**)&d_mask, Mask_Width * sizeof(float));
    checkCudaError("Failed to allocate device memory for mask");
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, Width * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("Failed to copy input to device");
    
    cudaMemcpy(d_mask, h_mask, Mask_Width * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("Failed to copy mask to device");
    
    // Launch kernel
    dim3 dimBlock(blockSize);
    dim3 dimGrid((Width + blockSize - 1) / blockSize);
    
    std::cout << "Launching kernel with grid=" << dimGrid.x << ", block=" << dimBlock.x << std::endl;
    
    conv1d_tiled_caching_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, d_mask, Mask_Width, Width);
    checkCudaError("Failed to launch kernel");
    
    cudaDeviceSynchronize();
    checkCudaError("Kernel execution failed");
    
    // Copy results back
    cudaMemcpy(h_output, d_output, Width * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("Failed to copy output from device");
    
    // Print results and verify
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Input:  ";
    for (int i = 0; i < Width; i++){
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Mask:   ";
    for (int i = 0; i < Mask_Width; i++){
        std::cout << h_mask[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "CPU:    ";
    for (int i = 0; i < Width; i++){
        std::cout << h_output_cpu[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "GPU:    ";
    for (int i = 0; i < Width; i++){
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify correctness
    bool correct = true;
    const float epsilon = 1e-5f;
    std::cout << "\n=== Verification ===" << std::endl;
    for (int i = 0; i < Width; i++){
        float diff = fabsf(h_output[i] - h_output_cpu[i]);
        if (diff > epsilon){
            std::cout << "Mismatch at index " << i << ": GPU=" << h_output[i] 
                      << ", CPU=" << h_output_cpu[i] << " (diff=" << diff << ")" << std::endl;
            correct = false;
        }
    }
    
    if (correct){
        std::cout << "✓ All results match! Kernel is correct." << std::endl;
    } else {
        std::cout << "✗ Results do not match!" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
    
    return correct ? 0 : 1;
}

