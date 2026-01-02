// flash attention forward pass
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>

__global__ void flashattention_forward(
    const float* __restrict__ Q, // [B, H, S, D], (batch_size, num_heads, seq_len, head_dim)
    const float* __restrict__ K, // [B, H, S, D]
    const float* __restrict__ V, // [B, H, S, D]
    float* __restrict__ O, // [B, H, S, D] output
    float* __restrict__ row_sum, // [B, H, S]
    float* __restrict__ row_max, // [B, H, S]
    int B, int H, int S, int D,
    int block_size, float scale
    )
{
    // Each block processes a (batch, head) pair.
    // Each thread in the block handles one token row q_idx
    int tid = threadIdx.x;
    int b = blockIdx.x;
    int h = blockIdx.y;

    extern __shared__ float shared_memory[];
    float*  k_tile = shared;                            // [block_size, D]
    float* v_tile = shared_memory + block_size * D;     // [block_size, D]
    
}

    