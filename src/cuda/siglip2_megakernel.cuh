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

}  // namespace siglip2_megakernel
