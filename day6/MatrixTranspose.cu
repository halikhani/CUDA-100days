#include <cuda_runtime.h>
#include <iostream>

// define size of the matrix
#define WIDTH 1024
#define HEIGHT 1024


__global__ void transposeMatrix(const float* input, float* output, int width, int height){
    // calcluate row and column of the thread

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // transpose if within bounds
    if (col < width && row < height){
        int inputIndex = row * width + col;
        int outputIndex = col * height + row;
        output[outputIndex] = input[inputIndex];
    }
}

// host function for checking cuda errors
void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << message << " - CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(){
    int width = WIDTH;
    int height = HEIGHT;

    // allocate host memory
    size_t size = width * height * sizeof(float);
    
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // Initialize the input matrix with some values
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(i);
    }

    // allocate device memory
    float* d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // copy data from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    checkCudaError("Failed to copy input data to device");

    // define block and grid sizes
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // launch the kernel
    transposeMatrix<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    checkCudaError("Failed to launch kernel");

    // copy data from device to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    checkCudaError("Failed to copy output data to host");

    // Verify the result
    bool success = true;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (h_output[i * height + j] != h_input[j * width + i]) {
                success = false;
                break;
            }
        }
    }

    std::cout << (success ? "Matrix transposition succeeded!" : "Matrix transposition failed!") << std::endl;

    // free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
    
}