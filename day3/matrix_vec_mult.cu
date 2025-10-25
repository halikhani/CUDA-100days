#include <iostream>
#include <cuda_runtime.h>

__global__ void matrix_vec_mult(float *A, float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i*N+j] * B[j];
        }
        C[i] = sum;
    }
}

int main()
{
    int N = 10;
    float *A, *B, *C;

    // initialize input matrices and vectors
    A = (float*)malloc(N*N*sizeof(float));
    B = (float*)malloc(N*sizeof(float));
    C = (float*)malloc(N*sizeof(float));
    
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0f;
        }
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N*N*sizeof(float));
    cudaMalloc((void**)&d_b, N*sizeof(float));
    cudaMalloc((void**)&d_c, N*sizeof(float));

    cudaMemcpy(d_a, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, C, N*sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size -1) / block_size;

    matrix_vec_mult<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}