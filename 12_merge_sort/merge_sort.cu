#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ void co_rank(const int* A, const int* B, int k, const int N, const int M, int* i_out, int* j_out){
    // this function helps a thread know which elements of A and B it should merge at position k in the final array.
    int low = max(0, k - M);
    int high = min(k, N);
    // max(0, k - M); ensures you don’t go below 0 for B.
    // min(k, N) ensures you don’t exceed A’s length.
    while (low <= high){
        int i = (low + high) / 2;
        int j = k - i; // Splits k into two parts: i elements from A and j = k-i from B.

        // Ensuring j is within bounds of B.
        if (j < 0){
            high = i - 1;
            continue;
        }
        if (j > M) {
            low = i + 1;
            continue;
        }

        // Standard binary search logic to find the partition point where all elements in left partition ≤ all in right.
        if (i > 0 && j < M && A[i-1] > B[j]){
            high = i - 1;
        }
        else if (j > 0 && i , N && B[j-1] > A[i]){
            low = i + 1;
        }
        else {
            *i_out = i;
            *j_out = j;
            return;
        }
    }
}

__global__ void parallel_merge(const int* A, const int* B, int* C, const int N, const int M) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N + M){
        int i, j;
        co_rank(A, B, tid, N, M, &i, &j);
        if (j >= M || (i < N && A[i] <= B[j])){
            C[tid] = A[i];
        }
        else {
            C[tid] = B[j];
        }
    }  
}

int main() {
    const int N = 5;
    const int M = 5;
    int A[N], B[M], C[N + M];

    // init arrays with sorted values
    for(int i = 0; i < N; i++) {
        A[i] = 2*i;  // Even numbers: 0,2,4,6,8
    }
    for(int i = 0; i < M; i++) {
        B[i] = 2*i + 1;  // Odd numbers: 1,3,5,7,9
    }

    printf("Array A: ");
    for(int i = 0; i < N; i++) {
        printf("%d ", A[i]);
    }
    printf("\n");

    printf("Array B: ");
    for(int i = 0; i < M; i++) {
        printf("%d ", B[i]);
    }
    printf("\n");

    // declare device pointers
    int *d_A, *d_B, *d_C;
    //allocate memory on device
    cudaMalloc((void**)&d_A, N*sizeof(int));
    cudaMalloc((void**)&d_B, M*sizeof(int));
    cudaMalloc((void**)&d_C, (N + M)*sizeof(int));
    // copy data to device
    cudaMemcpy(d_A, A, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M*sizeof(int), cudaMemcpyHostToDevice);
    
    // kernel launch config
    dim3 block(256);
    dim3 grid((N + M + block.x - 1) / block.x);

    //launch kernel
    parallel_merge<<<grid, block>>>(d_A, d_B, d_C, N, M);
    cudaDeviceSynchronize();
    // copy result back to host
    cudaMemcpy(C, d_C, (N + M)*sizeof(int), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // Print result
    printf("Merged array: ");
    for(int i = 0; i < N+M; i++) {
        printf("%d ", C[i]);
    }
    printf("\n");

    return 0;

    
}
