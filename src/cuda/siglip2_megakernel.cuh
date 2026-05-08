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

// QKV-prep, 2-kernel form. Replaces the (1 bias-add + 3 cont + 3 pad +
// 2 cast) = 9 launches per encoder block with 2 launches.
//
// Why 2 kernels: a single-kernel late-anchor design colliding with ggml's
// gallocr (qkv_add->data freed before V_cast.idx). The fix is to copy
// (mm + bias) into a persistent device scratch at qkv_add.idx, then split-
// permute-pad-cast from scratch at V_cast.idx — both anchors hit data that's
// guaranteed alive at their respective indices. See HANDOFF-megakernel-v0.md
// "Phase A1 — gallocr aliasing trap" for the full receipts.

// (1) Scratch-fill kernel — at qkv_add anchor: scratch[i] = mm[i] + bias[i%(3H)].
// Replaces the bias-add launch entirely; scratch will host the QKV biased
// output until the split kernel below consumes it (within the same CUDA
// stream, same encode, same block index — no cross-block lifetime issues
// because block N+1's qkv_add.idx > block N's V_cast.idx and the stream
// is sequential).
void launch_qkv_copy_to_scratch(
    const float *  mm,          // (3*H, n_pos)  F32 contiguous (mul_mat output)
    const float *  bias,        // (3*H,)        F32
    float *        scratch,     // (3*H, n_pos)  F32 contiguous (writeable)
    int            triH,        // 3 * H
    int            n_pos,
    cudaStream_t   stream);

// (2) Split-permute-pad-cast — at V_cast anchor: reads scratch (the biased
// qkv output) and writes the three FA inputs in one launch:
//   * Q  → q_pad   F32, ne=(d_pad, n_pos, n_head), contiguous
//   * K  → k_cast  F16, same shape
//   * V  → v_cast  F16, same shape
// Padded slots (d in [d_head, d_pad)) are written as exact-zero. Grid:
// (n_pos, 3*n_head). Block: (d_pad threads). One thread per output element.
void launch_fused_qkv_prep(
    const float *  qkv,         // (3*H, n_pos)  F32 contiguous (the scratch)
    float *        q_pad,       // (d_pad, n_pos, n_head)  F32
    void *         k_cast,      // (d_pad, n_pos, n_head)  F16  (raw __half *)
    void *         v_cast,      // (d_pad, n_pos, n_head)  F16
    int            H,
    int            n_pos,
    int            n_head,
    int            d_head,
    int            d_pad,
    cudaStream_t   stream);

// Phase A2: pointwise tail fusion. Fires per encoder block:
//   * P1 (bias + residual): y = (mm + bias_1d) + residual_2d. Anchored on the
//     OUTER add (the residual add); the inner bias-add is a follower. Saves
//     1 launch per anchor; per block this fires twice (o-proj and down-proj),
//     so 2 launches/block × 27 = 54/encode.
//   * P2 (bias + gelu): y = gelu(mm + bias_1d). Anchored on the GELU UNARY;
//     the upstream bias-add is a follower. Fires once per block (up-proj),
//     27/encode. GELU formula matches ggml's tanh approximation
//     (ggml_cuda_op_gelu_single) bit-for-bit.
//
// Both kernels assume 2D F32 contiguous mm/residual/output of shape (H, n_pos)
// with H innermost, plus a 1D F32 bias of length H broadcast over n_pos.
// __fadd_rn is used for the inner add to suppress nvcc's FMA fusion across
// the (mm+bias)→(+residual) chain, keeping per-store rounding identical to
// the split-kernel path. Drift is therefore reduction-tree noise only.
void launch_fused_bias_residual(
    const float *  mm,          // (H, n_pos)  F32 contiguous
    const float *  bias,        // (H,)        F32
    const float *  residual,    // (H, n_pos)  F32 contiguous (same shape as mm)
    float *        y,           // (H, n_pos)  F32 contiguous
    int            H,
    int            n_pos,
    cudaStream_t   stream);

void launch_fused_bias_gelu(
    const float *  mm,          // (H, n_pos)  F32 contiguous
    const float *  bias,        // (H,)        F32
    float *        y,           // (H, n_pos)  F32 contiguous
    int            H,
    int            n_pos,
    cudaStream_t   stream);

// Phase A3: post-FA cont fusion. After flash_attn_ext, fa_attn_pad_slice
// shrinks d-axis from d_pad → d_head and rebuilds a contiguous (H, n_pos)
// tensor (logically (d_head, n_head, n_pos) packed contiguous) for the o_proj
// mul_mat input. The current path is `view_3d(strided slice off d_pad tail) →
// ggml_cont` — one launch dispatched through ggml's generic cpy_f32_f32_kernel
// which carries arbitrary-stride parameter dispatch overhead. This kernel is
// the same operation specialized for these shapes (uniform contiguous output,
// uniform strided input over d_pad) so the per-launch path is shorter.
//
// One-launch operation; doesn't reduce kernel count vs ggml's cont, but the
// specialized path avoids ggml-cuda's generic-cpy dispatch tax and sets up
// the scaffolding for fusing the o_proj reads into this kernel later.
//
// Grid: (n_pos, n_head). Block: (d_head). One thread per output element.
void launch_fused_post_fa_cont(
    const float *  fa_out,      // (d_pad, n_head, n_pos)  F32 strided FA output
    float *        y,           // (d_head*n_head, n_pos)  F32 contiguous
    int            d_head,
    int            d_pad,
    int            n_head,
    int            n_pos,
    cudaStream_t   stream);

}  // namespace siglip2_megakernel
