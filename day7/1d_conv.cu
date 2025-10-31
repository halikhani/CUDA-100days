#include <cuda_runtime.h>
#include <iostream>

#define Mask_width 5
__constant__ float mask[Mask_width];

__global__ void conv1d(float* input, float* output, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n){
        float sum = 0.0f;
        for(int i = 0; i < Mask_width; i++){
            int input_idx = idx + i - Mask_width / 2;
            if(input_idx >= 0 && input_idx < n){
            sum += input[input_idx] * mask[Mask_width - i - 1];
            }
        }
        output[idx] = sum;
    }
}

// Host function to check for CUDA errors
void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << message << " - CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(){
    int n = 10;
    float h_input[n];
    float h_output[n];
    float h_M[Mask_width];

    for (int i=0; i < Mask_width;i++){
        h_M[i]=i;
    }

    for (int i=0; i<n;i++){
        h_input[i]=i;
    }

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, n * sizeof(float));
    checkCudaError("Failed to allocate device memory for input");
    cudaMalloc((void**)&d_output, n * sizeof(float));
    checkCudaError("Failed to allocate device memory for output");

    // Copy input to device
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("Failed to copy input data to device");

    // Copy mask to constant memory:
    cudaMemcpyToSymbol(mask, h_M, Mask_width * sizeof(float));
    checkCudaError("Failed to copy mask data to device");

    dim3 dimBlock(32);
    dim3 dimGrid((n + dimBlock.x - 1)/dimBlock.x);
    conv1d<<<dimGrid, dimBlock>>>(d_input, d_output, n);
    checkCudaError("Failed to launch kernel");
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("Failed to copy output data to host");
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    // print input 
    std::cout << "Input: ";
    for (int i=0; i<n;i++){
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    // print output
    std::cout << "Output: ";
    for (int i=0; i<n;i++){
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

    