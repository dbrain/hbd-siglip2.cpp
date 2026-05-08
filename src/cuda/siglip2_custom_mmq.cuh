// siglip2 custom MMQ — Phase C kernel-lap entry point.
//
// Specialized Q8_0 × F32 → F32 mul_mat for siglip2's exact shape set:
//
//   K = 1152                  (innermost; fixed across all 3 weights)
//   N ∈ { 1152, 3456, 4304 }  (attn_o, attn_qkv, ffn_up)
//   M ∈ { 64,   320,  729  }  (text n=1, text n=5 cfe, vision)
//
// This header is included only by siglip2_custom_mmq.cu and the microbench.
// The launcher takes raw Q8_0 weight + F32 activation and produces F32 output;
// it owns its own activation quantize-x pass into a caller-provided scratch
// (so the same scratch can be reused across multiple matmuls in one encode).
//
// Phase C step 1: a baseline mma m16n8k32 kernel that *matches* ggml's mma at
// attn_qkv M=320, K=1152, N=3456. If we can't match, we can't beat — abort
// early. After matching, specialize for the finite (M, N) set with
// compile-time K=1152 unroll, single Q8_0 layout, and (eventually) cp.async
// pipelining of the next K-block load over the current K-block mma.

#pragma once

#include <cstddef>
#include <cstdint>

// Forward-declare cudaStream_t so this header is includable from non-CUDA TUs
// (microbench, server, etc.) without dragging in <cuda_runtime.h>.
struct CUstream_st;
typedef CUstream_st * siglip2_cuda_stream_t;

namespace siglip2_custom_mmq {

// Activation scratch sizing helper. The kernel quantizes the F32 activation
// (K, M) into a Q8_1-equivalent layout: per-32-K block, 32 int8 quants packed
// as 8 ints + one f32 d (so the per-block float scale matches what an
// m16n8k32 INT8 mma path expects).
//
// Layout per block (one 32-K chunk for one M position):
//   [int8 qs[32]] = 8 ints (32 bytes)
//   [float    d ] = 1 int  (4 bytes)
// Total = 36 bytes per block. Per M position there are K/32 = 36 blocks.
// Per matmul activation total = M * 36 * 36 bytes for K=1152.
//
// Memory layout: (M, K_block, qs_or_d) — i.e. for a given m, the 36 blocks
// are contiguous and within each block qs[0..7] is followed by d. This makes
// per-thread coalesced reads in the mma path natural (one int per thread per
// k-step).
constexpr int K_FIXED       = 1152;
constexpr int K_BLOCK_SIZE  = 32;     // Q8_0 / Q8_1 block size
constexpr int K_BLOCKS      = K_FIXED / K_BLOCK_SIZE;  // 36
constexpr int QUANT_BYTES_PER_BLOCK = K_BLOCK_SIZE + sizeof(float); // 36

inline size_t activation_scratch_bytes(int M) {
    return (size_t) M * (size_t) K_BLOCKS * (size_t) QUANT_BYTES_PER_BLOCK;
}

// Quantize-x: F32 activation (K=1152, M) → packed int8+fp32-scale per 32-K
// block. Matches Q8_1's per-32-element max-abs / 127 scaling, with the float
// scale stored alongside qs (not as half2 like ggml's block_q8_1; for our
// usage we accumulate as scale_x * scale_w in F32 directly so a plain float
// is the natural format).
//
// Grid: (M, K_BLOCKS). Block: (32, 1, 1).  One warp per (M, K-block).
void launch_quantize_activation(
    const float * x,          // (K_FIXED, M) F32, K innermost
    void *        scratch,    // activation_scratch_bytes(M) bytes
    int           M,
    siglip2_cuda_stream_t stream);

// Per-block cycle counters emitted by the profiling kernel variant.
// 4 unsigned long long per CUDA block:
//   [0] = sum of cycles spent in load loops across the K-loop
//   [1] = sum of cycles spent in the post-load __syncthreads
//   [2] = sum of cycles spent in compute (scale-cache + ldmatrix A + per-M-sub
//         B-load + N_SUBTILES × mma + scale-and-add)
//   [3] = sum of cycles spent in the end-of-iter __syncthreads
struct ProfileCycles {
    unsigned long long load;
    unsigned long long sync1;
    unsigned long long compute;
    unsigned long long sync2;
};

// Custom Q8_0 × Q8_1 → F32 GEMM for siglip2 shapes.
//
// The weight tensor `w` is the raw GGUF Q8_0 buffer of shape (K, N) — i.e.
// rows-of-N each containing K_BLOCKS Q8_0 blocks (32 int8 quants + 1 fp16
// scale = 34 bytes / block). Standard ggml block_q8_0 layout, no repacking
// needed yet (Phase C step 1).
//
// `act_scratch` is the activation in Q8_1-equivalent form (see
// launch_quantize_activation).  `dst` is F32 of shape (N, M).
//
// Strides: row stride of w (in Q8_0 blocks) is K_BLOCKS = 36 (caller passes
// stride_w_row in bytes for safety vs raw Q8_0 blocks).  Output dst is
// row-major in N: dst[n + m*N] is the (n, m) entry.
void launch_custom_mmq(
    const void *  w,             // raw Q8_0 weights, shape (K, N)
    size_t        w_row_bytes,   // bytes per row-of-K (== K_BLOCKS * 34 for raw Q8_0)
    const void *  act_scratch,   // activation_scratch_bytes(M) packed
    float *       dst,           // (N, M) F32, N innermost
    int           M,
    int           N,
    siglip2_cuda_stream_t stream);

// Profiling variant: same kernel as launch_custom_mmq, but with clock64()-
// based per-region cycle counters written to `out_cycles`. The buffer must
// be sized for at least (n_block_n × n_block_m) ProfileCycles entries (one
// per CUDA block). Adds a few cycles of overhead per region; not for
// production. Returns the (n_block_n, n_block_m) grid dims via *out_grid_n
// and *out_grid_m so the caller can iterate the buffer.
void launch_custom_mmq_profile(
    const void *  w,
    size_t        w_row_bytes,
    const void *  act_scratch,
    float *       dst,
    int           M,
    int           N,
    void *        out_cycles,    // ProfileCycles * sized for (grid_n × grid_m)
    int *         out_grid_n,
    int *         out_grid_m,
    siglip2_cuda_stream_t stream);

}  // namespace siglip2_custom_mmq
