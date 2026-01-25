#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <numeric>
#include <cmath>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                         cudaGetErrorString(err), __FILE__, __LINE__);       \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

static int next_pow2(int v) {
    int p = 1;
    while (p < v) p <<= 1; // << for left shift (multiply by 2) and >> for right shift (divide by 2)
    return p;
}

// Brent-Kung scan per block (inclusive), power-of-two block size
__global__ void brent_kung_scan(const float* d_in, float* d_out, int n) {
    extern __shared__ float XY[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // load
    XY[tid] = (idx < n) ? d_in[idx] : 0.0f;
    __syncthreads();

    // upsweep
    // tree reduction in shared memory; each iteration combines pairs at distance stride to build partial sums.
    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        int index = (tid + 1) * 2 * stride - 1; // index of the current thread in the shared memory array, which is 2*stride elements away from the current thread.
        // 
        if (index < blockDim.x) {
            XY[index] += XY[index - stride];
        }
        __syncthreads();
    }

    // clear last element for exclusive scan (keeping last element as root)
    if (tid == blockDim.x - 1) {
        XY[tid] = 0.0f;
    }
    __syncthreads();

    // downsweep
    // in each iteration, left child gets parents value, and right child 
    // gets sum of original value of left child plus current value of parent
    for (unsigned int stride = blockDim.x >> 1; stride >= 1; stride >>= 1) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            float t = XY[index - stride]; // save original value of left child
            XY[index - stride] = XY[index]; // left child gets parents value
            XY[index] += t; // right child gets sum of original value of left child plus current value of parent
            // NOTE: since the tree is built to the right element of each pair during upsweep,
            // the index for right child is equal to index of the parent (index), 
            // and the index of the left child is index - stride.
        }
        __syncthreads();
        if (stride == 1) {
            break;
        }
    }

    // convert to inclusive for valid elements (add original value of the element to the prefix sum)
    if (idx < n) {
        d_out[idx] = XY[tid] + d_in[idx];
    }
}

int main() {
    const int n = 16;
    std::vector<float> h_in(n);
    for (int i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>(i + 1);
    }

    int block_size = next_pow2(n);
    const int max_block = 1024;
    if (block_size > max_block) {
        std::fprintf(stderr, "n too large for single block test\n");
        return 1;
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    brent_kung_scan<<<1, block_size, block_size * sizeof(float)>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::vector<float> h_ref(n);
    std::partial_sum(h_in.begin(), h_in.end(), h_ref.begin());

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_out[i] - h_ref[i]) > 1e-5f) {
            ok = false;
            break;
        }
    }

    std::printf("Input: ");
    for (int i = 0; i < n; ++i) std::printf("%.0f ", h_in[i]);
    std::printf("\nOutput: ");
    for (int i = 0; i < n; ++i) std::printf("%.0f ", h_out[i]);
    std::printf("\nResult: %s\n", ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return ok ? 0 : 1;
}