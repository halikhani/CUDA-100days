#include "cuda_kernels.h"
#include "helper_functions.h"

__global__ void addBiasKernel(float* output, const float* bias, int batch_size, int output_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = idx / output_features;
    int feature = idx % output_features;

    if (batch < batch_size && feature < output_features) {
        output[batch * output_features + feature] += bias[feature];
    }
}

void performLinearLayerOperation(
    cublasHandle_t cublas_handle,
    const float* input_data,
    const float* weights_data,
    const float* bias_data,
    float* output_data,
    int batch_size,
    int input_features,
    int output_features
){
    // This function orchestrates: Y = XW (using cuBLAS) and Y = Y + b (using CUDA kernel)
    // cuBLAS uses column-major, so for row-major data: output^T = weights^T * input^T
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // output^T = alpha * weights^T * input^T + beta * output^T
    checkCublasStatus(cublasSgemm(
        cublas_handle,
        CUBLAS_OP_T,     // transpose weights (weights^T)
        CUBLAS_OP_T,     // transpose input (input^T)
        output_features, // m: rows of weights^T and output^T
        batch_size,      // n: cols of input^T and output^T
        input_features,  // k: cols of weights^T, rows of input^T
        &alpha,
        weights_data,    // weights matrix (will be transposed)
        input_features,  // leading dim of weights (row-major: input_features)
        input_data,      // input matrix (will be transposed)
        input_features,  // leading dim of input (row-major: input_features)
        &beta,
        output_data,     // output matrix (stored as output^T in column-major)
        output_features  // leading dim of output (row-major: output_features)
    ));

    // bias addition
    int total_elements = batch_size * output_features;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    addBiasKernel<<<num_blocks, block_size>>>(output_data, bias_data, batch_size, output_features);
    checkCudaStatus(cudaGetLastError());
    checkCudaStatus(cudaDeviceSynchronize());

}

