#include <cstdio>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define SECTION_SIZE 256

__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, int InputSize){
    // each thread starts with one input element in the shared memory and repeatedly adds the value from a neighbor stride positions behind
    __shared__ float XY[SECTION_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < InputSize){
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }

    // code below performs iterative scan on XY (shared memory)
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        if (threadIdx.x >= stride) XY[threadIdx.x] += XY[threadIdx.x - stride];
    }

    if (i < InputSize){
        Y[i] = XY[threadIdx.x];
    }
    // to convert to exclusive scan, uncomment the lines below
    // if (i < InputSize && threadIdx.x != 0) {XY[threadIdx.x] = XY[threadIdx.x - 1];}
    // else {XY[threadIdx.x] = 0.0f;}

}

int main(){
    const int inputSize = 16;
    const int blockSize = 256;
    const int gridSize = (inputSize + blockSize - 1) / blockSize;

    std::vector<float> h_input(inputSize);
    std::vector<float> h_output(inputSize, 0.0f);
    std::vector<float> h_expected(inputSize, 0.0f);

    // initialize input
    for (int i = 0; i < inputSize; ++i){
        // static cast for converting int to float
        h_input[i] = static_cast<float>(i + 1);
    }

    // compute expected output
    float running = 0.0f;
    for (int i = 0; i < inputSize; ++i){
        running += h_input[i];
        h_expected[i] = running;
    }

    float *d_input = nullptr;
    float *d_output = nullptr;
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_output, inputSize * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice);
    Kogge_Stone_scan_kernel<<<gridSize, blockSize>>>(d_input, d_output, inputSize);
    cudaMemcpy(h_output.data(), d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < inputSize; ++i){
        // fabs is a floating point absolute value function
        if (std::fabs(h_output[i] - h_expected[i]) > 1e-5f){
            ok = false;
            std::printf("Mismatch at %d: got %.6f expected %.6f\n", i, h_output[i], h_expected[i]);
        }
    }

    if (ok){
        std::puts("Kogge-Stone scan test passed.");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return ok ? 0 : 1;
}