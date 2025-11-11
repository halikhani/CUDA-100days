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
    int tile_rows, int tile_cols, // number of query rows processed per block (e.g., 32), number of key cols processed per block (e.g., 32)
    float scale
    )
{
    extern __shared__ float shared[];
    float* q_tile = shared;
    float* k_tile = shared + tile_cols * D;
    float* v_tile = k_tile + tile_cols * D;
    float* score_tile = v_tile + tile_cols * D;
    //     [ q_tile ] [ k_tile ] [ v_tile ] [ score_tile ]
    //    ↓         ↓         ↓            ↓
    // [tc*D]    [tc*D]    [tc*D]       [tr*tc]
    int tid = threadIdx.x; // row in the tile to process
    int b = blockIdx.x; // batch index
    int h = blockIdx.y; // head index

    int qkv_offset = ((b * H + h) * S * D); 
    // think of b and h as an elem in a 2d array of size B*H, and then when we 
    // find the offset of it, we multiploy by S*D to get the offset of the qkv tensor
    int sum_offset = ((b * H + h) * S);
    // for the sum, there is no need to multiply by D, because we are only storing the sum and max for each row


    // Outer loop over tiles of query tokens (rows). Each block handles tile_rows query tokens at a time
    for (int row_block = 0; row_block < S; row_block += tile_rows) {
        // each thread processes one row (query token) from the tile.
        int q_row = row_block + tid;
        if (q_row >= S) return; // if the row is out of bounds, return (S is the seq_len)


        // Load Q[b, h, q_row, :] into a register tile for fast access (qkv_offset points to the start of Q[b, h, :, :])
        // note: here 64 is equal to D (head_dim), we presume its predefined in the kernel launch parameters
        float q_vec[64];
        for (int d = 0; d < D; ++d){
            q_vec[d] = Q[qkv_offset + q_row * D + d];
        }
        
    float max_val = -INFINITY;
    float sum_val = 0.0f;

    float o_vec[64] = {0.0f};

    // Loop over K/V tiles (column blocks) 
    // (Inner loop tiles across the key/value sequence, so each query row can attend over the full sequence in blocks)
    for (int col_block = 0; col_block < S; col_block += tile_cols) {
        //Load K and V tiles into shared memory:
        // Only the first tile_cols threads load keys and values (1 thread per row of K/V).
        if (tid < tile_cols){
            for (int d = 0; d < D; ++d){
                k_tile[tid * D + d] = K[qkv_offset + (col_block + tid) * D + d];
                v_tile[tid * D + d] = V[qkv_offset + (col_block + tid) * D + d];
            }
        }
        __syncthreads();




}

------------------------------------------------------------------------------------------------

        float o_vec[64] = {0.0f};

        for (int col_block = 0; col_block < S; col_block += tile_cols) {
            if (tid < tile_cols) {
                for (int d = 0; d < D; ++d) {
                    k_tile[tid * D + d] = K[qkv_offset + (col_block + tid) * D + d];
                    v_tile[tid * D + d] = V[qkv_offset + (col_block + tid) * D + d];
                }
            }
            __syncthreads();

            float scores[64];
            #pragma unroll
            for (int i = 0; i < tile_cols; ++i) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d)
                    dot += q_vec[d] * k_tile[i * D + d];
                dot *= scale;
                scores[i] = dot;
                if (dot > max_val) max_val = dot;
            }

            float local_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < tile_cols; ++i) {
                scores[i] = expf(scores[i] - max_val);
                local_sum += scores[i];
            }

            for (int d = 0; d < D; ++d) {
                for (int i = 0; i < tile_cols; ++i)
                    o_vec[d] += scores[i] * v_tile[i * D + d];
            }

            sum_val += local_sum;
            __syncthreads();
        }

        for (int d = 0; d < D; ++d)
            O[qkv_offset + q_row * D + d] = o_vec[d] / (sum_val + 1e-6f);
        row_sum[sum_offset + q_row] = sum_val;
        row_max[sum_offset + q_row] = max_val;
    }
}
