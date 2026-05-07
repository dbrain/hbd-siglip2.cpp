// siglip2 megakernel — internal CUDA declarations.
//
// Included only by siglip2_megakernel.cu (and by a future microbench, if we
// build one). Forward-declares ggml types we touch as opaque pointers; full
// definitions come from ggml-cuda.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend_cuda_context;

extern "C" {

// Hook signatures — must mirror ggml-cuda.cu's externs. We don't include
// ggml.h here to keep this file CUDA-only.

typedef void (*ggml_cuda_graph_begin_hook_fn)(
    ggml_backend_cuda_context * ctx,
    const ggml_cgraph *         cgraph);

typedef bool (*ggml_cuda_op_hook_fn)(
    ggml_backend_cuda_context * ctx,
    ggml_tensor *               dst,
    cudaStream_t                stream);

void ggml_cuda_set_graph_begin_hook(ggml_cuda_graph_begin_hook_fn fn);
void ggml_cuda_set_op_hook(ggml_cuda_op_hook_fn fn);

}  // extern "C"

namespace siglip2_megakernel {

// Fused LayerNorm-with-affine for 2D F32 tensors of shape (H, n_pos), where
// H is the innermost (contiguous) axis. Computes
//     mean    = sum(x_row) / H
//     var     = sum(x_row * x_row) / H - mean * mean
//     inv_std = rsqrtf(var + eps)
//     y[i]    = ((x[i] - mean) * inv_std) * w[i] + b[i]
// using the same reduction order as ggml's norm.cu (one-pass mean+var). FMA
// fusion on the final affine is suppressed via __fmul_rn / __fadd_rn so the
// result matches the (norm → mul → add) split path bit-for-bit modulo the
// reduction-tree shuffle, keeping cosine drift negligible across 82+ sites.
//
// Grid: (n_pos, 1, 1).  Block: BLOCK_DIM_X threads (1024 by default; falls
// back to WARP_SIZE for tiny H, mirroring ggml).
void launch_fused_layernorm_affine(
    const float *  x,           // (H, n_pos)  F32 contiguous
    const float *  w,           // (H,)        F32
    const float *  b,           // (H,)        F32
    float *        y,           // (H, n_pos)  F32 contiguous
    int            H,
    int            n_pos,
    float          eps,
    cudaStream_t   stream);

// Fused QKV-prep — replaces the (3 cont + 3 pad + 2 cast) chain that follows
// the qkv bias-add with one launch. Reads the strided F32 qkv buffer (3*H
// innermost, n_pos outer; ne[0] = 3*H is contiguous), splits along H into
// Q/K/V components, permutes (d_head, n_head, n_pos) → (d_pad, n_pos, n_head)
// while padding d_head→d_pad with zeros, and writes:
//   * Q  → q_pad   F32, ne=(d_pad, n_pos, n_head), contiguous
//   * K  → k_cast  F16, same shape
//   * V  → v_cast  F16, same shape
//
// d_pad must be ≥ d_head and may equal d_head (no pad). Padded slots are
// written exact-zero (no denormals). Grid: (n_pos, 3*n_head). Block:
// (d_pad threads). One thread per output element across all three outputs.
void launch_fused_qkv_prep(
    const float *  qkv,         // (3*H, n_pos)  F32 contiguous
    float *        q_pad,       // (d_pad, n_pos, n_head)  F32
    void *         k_cast,      // (d_pad, n_pos, n_head)  F16  (raw __half *)
    void *         v_cast,      // (d_pad, n_pos, n_head)  F16
    int            H,
    int            n_pos,
    int            n_head,
    int            d_head,
    int            d_pad,
    cudaStream_t   stream);

}  // namespace siglip2_megakernel
