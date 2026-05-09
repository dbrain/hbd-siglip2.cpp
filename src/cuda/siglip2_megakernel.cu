// siglip2 megakernel — Phase A0: fused LayerNorm-with-affine.
//
// Detects every (ggml_norm → ggml_mul-with-weight → ggml_add-with-bias) chain
// at graph-begin time, fires one CUDA kernel per chain (vs ggml's 3 separate
// kernels), and short-circuits the upstream norm and mul ops via the per-op
// hook so they don't execute.
//
// Per text encode (n_pos=64, 27 blocks): 55 LN sites × 2 ops fused = 110
// launches saved. Per vision encode (n_pos=729): 56 LN sites × 2 = 112. Per
// /v1/classify (vision + 5 texts): 56 + 5×55 = 331 sites × 2 = 662 saved.
//
// Numerical: same one-pass mean+var formula as ggml/src/ggml-cuda/norm.cu;
// affine compute uses __fmul_rn / __fadd_rn to suppress FMA fusion so the
// final value matches the split (norm → mul → add) path's per-store rounding
// bit-for-bit (modulo the reduction-tree shuffle, which is identical thread
// layout for matching block size).
//
// Kill switch: SIGLIP2_DISABLE_MEGAKERNEL=1 in the env at boot. Verbose
// boot logging: SIGLIP2_MEGAKERNEL_VERBOSE=1.

#include "siglip2_megakernel.h"
#include "siglip2_megakernel.cuh"

#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace siglip2_megakernel {

// ----------------------------------------------------------------------------
// Fused LayerNorm-with-affine kernel
// ----------------------------------------------------------------------------

namespace {

constexpr int WARP_SIZE = 32;
constexpr int LN_BLOCK  = 1024;  // 32 warps per block

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off, WARP_SIZE);
    }
    return v;
}

// Two-axis warp/block reduction matching ggml/src/ggml-cuda/norm.cu's
// (sum, sum_sq) float2 layout. Same shared-mem cross-warp combine.
template <int BLOCK_SIZE>
__device__ __forceinline__ float2 block_reduce_sum2(float2 v, float2 * smem) {
    v.x = warp_reduce_sum(v.x);
    v.y = warp_reduce_sum(v.y);
    constexpr int N_WARPS = BLOCK_SIZE / WARP_SIZE;
    if constexpr (N_WARPS > 1) {
        const int lane = threadIdx.x & (WARP_SIZE - 1);
        const int wid  = threadIdx.x >> 5;
        if (lane == 0) smem[wid] = v;
        __syncthreads();
        if (wid == 0) {
            float2 acc = (threadIdx.x < N_WARPS) ? smem[threadIdx.x] : make_float2(0.0f, 0.0f);
            acc.x = warp_reduce_sum(acc.x);
            acc.y = warp_reduce_sum(acc.y);
            if (lane == 0) smem[0] = acc;
        }
        __syncthreads();
        v = smem[0];
    }
    return v;
}

// Fused norm + affine. Grid: (n_pos, 1, 1). One block per row.
template <int BLOCK_SIZE>
__global__ void fused_layernorm_affine_kernel(
        const float * __restrict__ x,
        const float * __restrict__ w,
        const float * __restrict__ b,
        float       * __restrict__ y,
        const int                  H,
        const float                eps) {
    const int  pos    = blockIdx.x;
    const int  tid    = threadIdx.x;
    const float * xrow = x + pos * H;
    float       * yrow = y + pos * H;

    float2 acc = make_float2(0.0f, 0.0f);
    for (int i = tid; i < H; i += BLOCK_SIZE) {
        const float v = xrow[i];
        acc.x += v;
        acc.y += v * v;
    }

    __shared__ float2 smem[BLOCK_SIZE / WARP_SIZE];
    acc = block_reduce_sum2<BLOCK_SIZE>(acc, smem);

    const float mean    = acc.x / (float) H;
    const float var     = acc.y / (float) H - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    // Affine: ((x - mean) * inv_std) * w + b. We split the multiplies
    // explicitly with __fmul_rn / __fadd_rn so nvcc cannot FMA-fuse them
    // — that keeps each output bit-identical to ggml's split (norm → mul
    // → add) per-store-rounding path. Without this, fmul+add fuses to FMA
    // (one rounding) and ggml does two roundings (norm→dst, mul→dst, add→dst
    // — the second-and-third are reads of intermediate F32 that are stable),
    // producing a 0.5-ulp drift per output × 1152 outputs × 82+ sites that
    // can compound past the 0.999 cosine floor. Defensive.
    for (int i = tid; i < H; i += BLOCK_SIZE) {
        const float t  = (xrow[i] - mean) * inv_std;
        const float tw = __fmul_rn(t, w[i]);
        yrow[i] = __fadd_rn(tw, b[i]);
    }
}

}  // namespace

void launch_fused_layernorm_affine(
        const float * x,
        const float * w,
        const float * b,
        float       * y,
        int           H,
        int           n_pos,
        float         eps,
        cudaStream_t  stream) {
    const dim3 grid((unsigned) n_pos, 1, 1);
    const dim3 block((unsigned) LN_BLOCK, 1, 1);
    fused_layernorm_affine_kernel<LN_BLOCK><<<grid, block, 0, stream>>>(
        x, w, b, y, H, eps);
}

// ----------------------------------------------------------------------------
// QKV-prep — 2-kernel form (copy-to-scratch + split-permute-pad-cast)
// ----------------------------------------------------------------------------
//
// Replaces (3 cont + 3 pad + 2 cast) per encoder block with one launch.
//
// Source layout — qkv is the bias-add output, F32 contiguous, ne=(3*H, n_pos)
// with ne[0]=3*H innermost. For position p, head h, component c (0=Q, 1=K,
// 2=V), the d_head-element vector lives at byte offset
//   p * (3*H * 4) + c * H * 4 + h * d_head * 4
// i.e. flat F32 index `p * 3*H + c * H + h * d_head + d`.
//
// Destination layout — each of {q_pad, k_cast, v_cast} is contiguous in the
// permuted/padded shape ne=(d_pad, n_pos, n_head) where ne[0]=d_pad is
// innermost, ne[1]=n_pos, ne[2]=n_head. Flat index `h * d_pad * n_pos +
// p * d_pad + d`. q_pad is F32; k_cast / v_cast are F16. Padded slots
// (d in [d_head, d_pad)) are written as exact zero.
//
// Grid: (n_pos, 3*n_head, 1). Block: (d_pad, 1, 1). One thread per output
// element across all three outputs.
namespace {

// (1) Scratch-fill: scratch[i] = mm[i] + bias[i % triH].
//     Grid: (n_pos, ceil(triH / 256), n_batch). Block: (256, 1, 1).
//     The batch dim is independent; per-batch the layout is exactly the
//     unbatched (3*H, n_pos) F32 contiguous block.
__global__ void qkv_copy_to_scratch_kernel(
        const float * __restrict__ mm,
        const float * __restrict__ bias,
        float       * __restrict__ scratch,
        const int                  triH,
        const int                  n_pos) {
    const int p   = blockIdx.x;
    const int tid = blockIdx.y * blockDim.x + threadIdx.x;
    if (tid >= triH) return;
    const int b   = blockIdx.z;
    const int per_batch = triH * n_pos;
    const int idx = b * per_batch + p * triH + tid;
    scratch[idx] = mm[idx] + bias[tid];
}

__global__ void fused_qkv_prep_kernel(
        const float * __restrict__ qkv,
        float       * __restrict__ q_pad,
        __half      * __restrict__ k_cast,
        __half      * __restrict__ v_cast,
        const int                  H,
        const int                  n_head,
        const int                  d_head,
        const int                  d_pad,
        const int                  n_pos) {
    const int d  = threadIdx.x;
    if (d >= d_pad) return;
    const int p  = blockIdx.x;
    const int hc = blockIdx.y;          // 0..3*n_head - 1
    const int c  = hc / n_head;         // 0=Q, 1=K, 2=V
    const int h  = hc - c * n_head;
    const int b  = blockIdx.z;

    const int qkv_per_batch = 3 * H * n_pos;
    const int dst_per_batch = d_pad * n_pos * n_head;
    const float * __restrict__ qkv_b   = qkv    + b * qkv_per_batch;
    float       * __restrict__ q_pad_b = q_pad  + b * dst_per_batch;
    __half      * __restrict__ k_cast_b = k_cast + b * dst_per_batch;
    __half      * __restrict__ v_cast_b = v_cast + b * dst_per_batch;

    const int dst_idx = h * d_pad * n_pos + p * d_pad + d;

    if (d < d_head) {
        const int src_idx = p * 3 * H + c * H + h * d_head + d;
        const float v = qkv_b[src_idx];
        if (c == 0) {
            q_pad_b[dst_idx] = v;
        } else if (c == 1) {
            k_cast_b[dst_idx] = __float2half(v);
        } else {
            v_cast_b[dst_idx] = __float2half(v);
        }
    } else {
        // Pad slot — exact zero (no denormals).
        if (c == 0) {
            q_pad_b[dst_idx] = 0.0f;
        } else if (c == 1) {
            k_cast_b[dst_idx] = __ushort_as_half((unsigned short) 0);
        } else {
            v_cast_b[dst_idx] = __ushort_as_half((unsigned short) 0);
        }
    }
}

}  // namespace

void launch_qkv_copy_to_scratch(
        const float * mm,
        const float * bias,
        float       * scratch,
        int           triH,
        int           n_pos,
        int           n_batch,
        cudaStream_t  stream) {
    constexpr int TPB = 256;
    const dim3 grid((unsigned) n_pos,
                    (unsigned) ((triH + TPB - 1) / TPB),
                    (unsigned) n_batch);
    const dim3 block((unsigned) TPB, 1, 1);
    qkv_copy_to_scratch_kernel<<<grid, block, 0, stream>>>(mm, bias, scratch, triH, n_pos);
}

void launch_fused_qkv_prep(
        const float * qkv,
        float       * q_pad,
        void        * k_cast,
        void        * v_cast,
        int           H,
        int           n_pos,
        int           n_head,
        int           d_head,
        int           d_pad,
        int           n_batch,
        cudaStream_t  stream) {
    const dim3 grid((unsigned) n_pos, (unsigned) (3 * n_head), (unsigned) n_batch);
    const dim3 block((unsigned) d_pad, 1, 1);
    fused_qkv_prep_kernel<<<grid, block, 0, stream>>>(
        qkv, q_pad,
        static_cast<__half *>(k_cast),
        static_cast<__half *>(v_cast),
        H, n_head, d_head, d_pad, n_pos);
}

// ----------------------------------------------------------------------------
// Phase A2: pointwise tail fusion (bias+residual, bias+gelu)
// ----------------------------------------------------------------------------

namespace {

// Match ggml's gelu tanh-approximation (ggml_cuda_op_gelu_single) bit-for-bit.
__device__ __forceinline__ float fused_gelu_single(float x) {
    constexpr float GELU_COEF_A    = 0.044715f;
    constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

// y[idx] = (mm[idx] + bias[idx % H]) + residual[idx]
// The two adds are written with __fadd_rn so nvcc cannot FMA-fuse them; this
// preserves per-store F32 rounding parity with ggml's split (bias_add → add)
// path. Bias-broadcast uses a divrem since H may not be a power of two
// (H = d_head * n_head, e.g. 1152 for siglip2).
__global__ void fused_bias_residual_kernel(
        const float * __restrict__ mm,
        const float * __restrict__ bias,
        const float * __restrict__ residual,
        float       * __restrict__ y,
        const int                  H,
        const int                  n_pos) {
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = H * n_pos;
    if (idx >= total) return;
    const int h = idx % H;
    const float mb = __fadd_rn(mm[idx], bias[h]);
    y[idx] = __fadd_rn(mb, residual[idx]);
}

// y[idx] = gelu(mm[idx] + bias[idx % H])
__global__ void fused_bias_gelu_kernel(
        const float * __restrict__ mm,
        const float * __restrict__ bias,
        float       * __restrict__ y,
        const int                  H,
        const int                  n_pos) {
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = H * n_pos;
    if (idx >= total) return;
    const int h = idx % H;
    const float mb = __fadd_rn(mm[idx], bias[h]);
    y[idx] = fused_gelu_single(mb);
}

}  // namespace

void launch_fused_bias_residual(
        const float * mm,
        const float * bias,
        const float * residual,
        float       * y,
        int           H,
        int           n_pos,
        cudaStream_t  stream) {
    constexpr int TPB = 256;
    const int total = H * n_pos;
    const dim3 grid((unsigned) ((total + TPB - 1) / TPB), 1, 1);
    const dim3 block((unsigned) TPB, 1, 1);
    fused_bias_residual_kernel<<<grid, block, 0, stream>>>(mm, bias, residual, y, H, n_pos);
}

void launch_fused_bias_gelu(
        const float * mm,
        const float * bias,
        float       * y,
        int           H,
        int           n_pos,
        cudaStream_t  stream) {
    constexpr int TPB = 256;
    const int total = H * n_pos;
    const dim3 grid((unsigned) ((total + TPB - 1) / TPB), 1, 1);
    const dim3 block((unsigned) TPB, 1, 1);
    fused_bias_gelu_kernel<<<grid, block, 0, stream>>>(mm, bias, y, H, n_pos);
}

// ----------------------------------------------------------------------------
// Phase A3: post-FA cont fusion (strided slice + contiguous copy)
// ----------------------------------------------------------------------------

namespace {

// Per batch b in [0, n_batch):
//   src[(d, h, p, b)] = fa_out[b * (d_pad*n_head*n_pos) + d + h*d_pad + p*d_pad*n_head]
//   dst[(d, h, p, b)] = y     [b * (d_head*n_head*n_pos) + d + h*d_head + p*d_head*n_head]
// One thread per output element. Block over d_head, grid over (n_pos, n_head, n_batch).
__global__ void fused_post_fa_cont_kernel(
        const float * __restrict__ fa_out,
        float       * __restrict__ y,
        const int                  d_head,
        const int                  d_pad,
        const int                  n_head,
        const int                  n_pos) {
    const int d = threadIdx.x;
    if (d >= d_head) return;
    const int h = blockIdx.y;
    const int p = blockIdx.x;
    const int b = blockIdx.z;
    const int H = d_head * n_head;

    const int src_per_batch = d_pad  * n_head * n_pos;
    const int dst_per_batch = d_head * n_head * n_pos;
    const int src_idx = b * src_per_batch + d + h * d_pad + p * d_pad * n_head;
    const int dst_idx = b * dst_per_batch + d + h * d_head + p * H;
    y[dst_idx] = fa_out[src_idx];
}

}  // namespace

void launch_fused_post_fa_cont(
        const float * fa_out,
        float       * y,
        int           d_head,
        int           d_pad,
        int           n_head,
        int           n_pos,
        int           n_batch,
        cudaStream_t  stream) {
    const dim3 grid((unsigned) n_pos, (unsigned) n_head, (unsigned) n_batch);
    const dim3 block((unsigned) d_head, 1, 1);
    fused_post_fa_cont_kernel<<<grid, block, 0, stream>>>(
        fa_out, y, d_head, d_pad, n_head, n_pos);
}

// ----------------------------------------------------------------------------
// Plan: (norm → mul-with-weight → add-with-bias) chain detection
// ----------------------------------------------------------------------------
//
// Anchor = the ADD node (final dst is what downstream ops read). At graph-
// begin time we walk every ADD node, check the (src[0]=MUL whose src[0]=NORM)
// pattern, validate shapes (2D F32 contiguous, weight + bias 1D F32 of length
// ne[0]), and record:
//
//   anchors[add_node] = { x = norm.src[0], w = mul.src[1], b = add.src[1],
//                         H = ne[0], n_pos = ne[1], eps = norm op_params[0] }
//   followers.insert(norm_node);
//   followers.insert(mul_node);
//
// op_hook(dst):
//   - dst in anchors    → fire fused kernel writing to dst->data, return true
//   - dst in followers  → return true (no-op; norm's and mul's data are
//                         garbage but no downstream op reads them)
//   - else              → return false (ggml's normal dispatch handles it)
//
// Plan is rebuilt per graph compute. Globals are safe because every call into
// ggml_backend_sched_graph_compute is serialized by siglip2_server's
// encode_mutex.

struct LayerNormEntry {
    const ggml_tensor * x_node    = nullptr;
    const ggml_tensor * w_node    = nullptr;
    const ggml_tensor * b_node    = nullptr;
    const ggml_tensor * norm_node = nullptr;
    const ggml_tensor * mul_node  = nullptr;
    int   H     = 0;
    int   n_pos = 0;
    float eps   = 1e-6f;
};

// ----------------------------------------------------------------------------
// Phase A1: QKV-prep chain plan
// ----------------------------------------------------------------------------
//
// One QkvPrepEntry per encoder block whose qkv bias-add is followed by the
// (3 cont + 3 pad + 2 cast) chain feeding flash_attn_ext. Anchor = whichever
// of {q_pad, k_cast, v_cast} has the highest topo index in the cgraph (so
// when the op-hook anchors and the fused kernel writes all three dsts, the
// upstream followers' op_hook calls have already returned true).

// Phase A2 entries — pointwise tail fusion. Anchor on the OUTER op (residual
// add or GELU); inner bias-add is the follower. Read the full design notes
// above launch_fused_bias_residual / launch_fused_bias_gelu in the .cuh.
//
// Gallocr-trap caveat: skipping the bias-add follower only works if gallocr
// packs mm.dst, bias_add.dst, and outer.dst into the same physical slot, so
// the slot still holds mm's value when our late anchor reads it. This holds
// in the per-encoder-block sub-graph (long, repetitive, shape-uniform), but
// breaks for one-off chains like the vision probe head (gelu reads garbage)
// and the patch embedding (non-consecutive). The plan-builder gates on both
// `is_blk_bias` (bias name contains "blk") and consecutive-in-topo to scope
// the fusion to the safe region.

struct BiasResidualEntry {
    const ggml_tensor * mm_node       = nullptr;  // mul_mat output  (H, n_pos)
    const ggml_tensor * bias_node     = nullptr;  // 1D F32 bias     (H,)
    const ggml_tensor * residual_node = nullptr;  // 2D F32 residual (H, n_pos)
    const ggml_tensor * inner_add     = nullptr;  // follower
    int H     = 0;
    int n_pos = 0;
};

struct BiasGeluEntry {
    const ggml_tensor * mm_node   = nullptr;
    const ggml_tensor * bias_node = nullptr;
    const ggml_tensor * inner_add = nullptr;  // follower
    int H     = 0;
    int n_pos = 0;
};

// Phase A3: post-FA cont. Anchor on the cont node; follower = nothing (the
// upstream view is metadata-only, no compute). The cont currently reads a
// strided 3D F32 view (slicing d_head off the d_pad tail of FA's output) and
// writes contiguous (d_head, n_head, n_pos). Our kernel does the same memory
// pattern with shape-specialized indexing.
struct PostFaContEntry {
    const ggml_tensor * fa_out_node = nullptr;  // FA output (the 3D view's parent)
    int d_head  = 0;
    int d_pad   = 0;
    int n_head  = 0;
    int n_pos   = 0;
    int n_batch = 1;
};

struct QkvPrepEntry {
    // Copy-to-scratch fields (used at qkv_add anchor)
    const ggml_tensor * mm_node     = nullptr;  // mul_mat output (F32, 3*H × n_pos)
    const ggml_tensor * bias_node   = nullptr;  // qkv bias (F32, 3*H)
    const ggml_tensor * qkv_add     = nullptr;  // anchor 1 — bias-add node
    // Split-from-scratch fields (used at v_cast / split anchor)
    const ggml_tensor * q_pad_dst   = nullptr;
    const ggml_tensor * k_cast_dst  = nullptr;
    const ggml_tensor * v_cast_dst  = nullptr;
    const ggml_tensor * split_anchor = nullptr; // anchor 2 — last of {q_pad,k_cast,v_cast}
    int H       = 0;
    int n_pos   = 0;
    int n_head  = 0;
    int d_head  = 0;
    int d_pad   = 0;
    int n_batch = 1;
};

// ─── Per-anchor op profiling state ──────────────────────────────────────────
//
// SIGLIP2_PROFILE_OPS=1 → record a cudaEvent at every megakernel anchor fire
// (before the fused launch). With CUDA graphs on, those records get captured
// into the graph and re-fire on every replay with fresh timestamps. After
// graph_compute returns, profile_after_encode() syncs the last event, walks
// the recorded ranges, and dumps an aggregate breakdown by anchor type.
//
// Event pool is monotonically-grown per-thread; each encode resets the index
// to 0 and reuses the same event objects for the same logical position. The
// captured graph references those exact event objects, so timing data stays
// consistent across replays of the same shape.
//
// Tags:
//   0 = LN, 1 = QKV-copy, 2 = QKV-split, 3 = post-FA cont,
//   4 = tail-bias+residual, 5 = tail-bias+gelu, -1 = end-marker
constexpr int kProfTagLN        = 0;
constexpr int kProfTagQkvCopy   = 1;
constexpr int kProfTagQkvSplit  = 2;
constexpr int kProfTagPostFA    = 3;
constexpr int kProfTagTailResid = 4;
constexpr int kProfTagTailGelu  = 5;
constexpr int kProfTagEnd       = 6;
constexpr int kProfNumTags      = 7;

struct ProfileState {
    bool                          enabled = false;
    bool                          checked = false;
    std::vector<cudaEvent_t>      events;
    std::vector<int>              tags;
    size_t                        used    = 0;
};

thread_local ProfileState g_profile;

bool profile_enabled_check() {
    if (!g_profile.checked) {
        g_profile.checked = true;
        const char * v = std::getenv("SIGLIP2_PROFILE_OPS");
        g_profile.enabled = v && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }
    return g_profile.enabled;
}

inline void profile_record_inline(int tag, cudaStream_t stream) {
    if (!profile_enabled_check()) return;
    if (g_profile.used >= g_profile.events.size()) {
        cudaEvent_t e = nullptr;
        if (cudaEventCreate(&e) != cudaSuccess) return;
        g_profile.events.push_back(e);
        g_profile.tags.push_back(tag);
    } else {
        g_profile.tags[g_profile.used] = tag;
    }
    cudaEventRecord(g_profile.events[g_profile.used], stream);
    ++g_profile.used;
}

// QKV-prep scratch — provided by the encoder, registered per host thread via
// siglip2_megakernel::set_active_qkv_scratch before each graph_compute. Each
// encoder owns one slab pre-allocated to its worst-case (3*H, n_pos_max,
// n_batch_max) at load() time; the slab lives as long as the encoder's
// captured CUDA graphs do. Per-thread storage handles the dual-tower
// concurrency in /v1/classify (vision + text run on different OS threads
// with their own private streams) without any global pool.
//
// Falling back is graceful: if the active slab is too small for the current
// encode (n_batch larger than what the encoder pre-sized for, etc.) the
// QKV-prep anchors return false and ggml's split path runs as if the
// fusion were disabled.
thread_local void *  g_active_qkv_scratch_dptr = nullptr;
thread_local size_t  g_active_qkv_scratch_cap  = 0;

namespace {

// All anchor + follower maps are thread_local because /v1/classify runs
// vision encode + text encode concurrently on private streams — each
// encode owns one host thread, both call graph_begin_hook → build_*_plan
// (which clears + rebuilds), then op_hook (which reads). Globals would
// race on clear/insert from two threads at once and corrupt the bucket
// structure (deterministic SIGSEGV on first /v1/classify after /unload,
// non-deterministic glitches under load). Per-thread maps eliminate the
// race because each compute is single-threaded end-to-end.
thread_local std::unordered_map<const ggml_tensor *, LayerNormEntry> g_ln_anchors;
thread_local std::unordered_set<const ggml_tensor *>                 g_ln_followers;

// Two anchor maps: one for the copy-to-scratch kernel (anchored on
// qkv_add), one for the split-from-scratch kernel (anchored on the last
// of {q_pad, k_cast, v_cast}). Plan-builder populates both with the same
// QkvPrepEntry so the dispatcher reads from one map per anchor lookup.
thread_local std::unordered_map<const ggml_tensor *, QkvPrepEntry>   g_qkv_copy_anchors;
thread_local std::unordered_map<const ggml_tensor *, QkvPrepEntry>   g_qkv_split_anchors;
thread_local std::unordered_set<const ggml_tensor *>                 g_qkv_followers;

thread_local std::unordered_map<const ggml_tensor *, BiasResidualEntry> g_br_anchors;
thread_local std::unordered_map<const ggml_tensor *, BiasGeluEntry>     g_bg_anchors;
thread_local std::unordered_set<const ggml_tensor *>                    g_tail_followers;

thread_local std::unordered_map<const ggml_tensor *, PostFaContEntry>   g_post_fa_anchors;

bool     g_enabled              = true;
// QKV-prep — 2-kernel form: anchor 1 on qkv_add copies mm+bias into a
// persistent device-side scratch buffer (replacing the bias-add); anchor 2
// on the topo-last of {q_pad, k_cast, v_cast} split-permute-pad-casts from
// scratch into the FA inputs. 9 ops per block (1 add + 3 cont + 3 pad +
// 2 cast) → 2 kernels. Bit-identical to ggml's split path. Kill switch:
// SIGLIP2_DISABLE_QKV_PREP=1. The earlier single-kernel design hit a
// gallocr aliasing trap (qkv_add slot reclaimed before the late anchor
// fired) — the two-kernel split is the workaround.
bool     g_qkv_enabled          = true;
// Phase A2 — pointwise tail fusion: (mm + 1D bias) + 2D residual collapsed
// into one launch (P1, fires at o-proj and down-proj per block), and gelu(mm
// + 1D bias) collapsed into one launch (P2, fires at up-proj per block). 3
// launches/block × 27 blocks = 81/encode. Anchor on OUTER op (residual_add
// or GELU); inner bias_add is the follower. Bit-clean against the split path
// (both kernels use __fadd_rn for the inner add to suppress nvcc FMA fusion;
// GELU formula matches ggml_cuda_op_gelu_single). Kill switch:
// SIGLIP2_DISABLE_TAIL_FUSION=1.
bool     g_tail_enabled         = true;
// Phase A3 — post-FA cont. Strided 3D view (slicing d_pad → d_head off FA's
// output) + ggml_cont collapsed into one specialized kernel. Same launch
// count (1) as the existing cont, but skips ggml-cuda's generic-cpy dispatch
// path. Default ON; SIGLIP2_DISABLE_POST_FA_CONT=1 to fall back to ggml's cpy.
bool     g_post_fa_enabled      = true;
// Per-thread counters paired with the per-thread anchor maps. Diagnostics
// printed via SIGLIP2_MEGAKERNEL_VERBOSE only see the calling thread's
// counts — fine, they're best-effort logging, not metrics.
thread_local uint64_t g_ln_groups_built      = 0;
thread_local uint64_t g_ln_anchor_fires      = 0;
thread_local uint64_t g_ln_follower_fires    = 0;
thread_local uint64_t g_qkv_groups_built     = 0;
thread_local uint64_t g_qkv_anchor_fires     = 0;
thread_local uint64_t g_qkv_follower_fires   = 0;
thread_local uint64_t g_br_groups_built      = 0;
thread_local uint64_t g_bg_groups_built      = 0;
thread_local uint64_t g_tail_anchor_fires    = 0;
thread_local uint64_t g_tail_follower_fires  = 0;
thread_local uint64_t g_post_fa_groups_built = 0;
thread_local uint64_t g_post_fa_anchor_fires = 0;
int      g_log_budget           = 0;

bool is_2d_f32_contiguous(const ggml_tensor * t) {
    if (!t) return false;
    if (t->type != GGML_TYPE_F32) return false;
    if (t->ne[2] != 1 || t->ne[3] != 1) return false;
    if (t->nb[0] != ggml_type_size(t->type)) return false;
    if (t->nb[1] != t->nb[0] * t->ne[0])     return false;
    return true;
}

// Permissive variant for pointwise-across-positions kernels (LN, bias+residual,
// bias+gelu) that don't care about per-batch boundaries — they iterate flat
// across all positions × batches as long as memory is contiguous along H.
// Allows ne[2]>1 (the prompt-batch dim from encode_batch). Treats the effective
// row count as ne[1]*ne[2].
bool is_2d_or_3d_f32_contiguous(const ggml_tensor * t) {
    if (!t) return false;
    if (t->type != GGML_TYPE_F32) return false;
    if (t->ne[3] != 1) return false;
    if (t->nb[0] != ggml_type_size(t->type)) return false;
    if (t->nb[1] != t->nb[0] * t->ne[0])     return false;
    if (t->ne[2] > 1 && t->nb[2] != t->nb[1] * t->ne[1]) return false;
    return true;
}

int64_t flat_n_pos(const ggml_tensor * t) {
    return t->ne[1] * t->ne[2];
}

bool is_1d_f32_affine(const ggml_tensor * t, int64_t H) {
    if (!t) return false;
    if (t->type != GGML_TYPE_F32) return false;
    if (t->ne[0] != H) return false;
    // Affine vectors are broadcast over the n_pos axis: ne[1..3] = 1.
    if (t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1) return false;
    return true;
}

void build_ln_plan(const ggml_cgraph * cgraph) {
    g_ln_anchors.clear();
    g_ln_followers.clear();

    if (!cgraph) return;
    const int n = ggml_graph_n_nodes((ggml_cgraph *) cgraph);
    for (int i = 0; i < n; ++i) {
        ggml_tensor * add = ggml_graph_node((ggml_cgraph *) cgraph, i);
        if (!add || add->op != GGML_OP_ADD) continue;
        ggml_tensor * mul = add->src[0];
        ggml_tensor * b   = add->src[1];
        if (!mul || !b)               continue;
        if (mul->op != GGML_OP_MUL)   continue;
        ggml_tensor * norm = mul->src[0];
        ggml_tensor * w    = mul->src[1];
        if (!norm || !w)              continue;
        if (norm->op != GGML_OP_NORM) continue;
        ggml_tensor * x    = norm->src[0];
        if (!x)                       continue;

        if (!is_2d_or_3d_f32_contiguous(x))                  continue;
        if (!is_2d_or_3d_f32_contiguous(norm))               continue;
        if (!is_2d_or_3d_f32_contiguous(mul))                continue;
        if (!is_2d_or_3d_f32_contiguous(add))                continue;
        if (x->ne[1]    != norm->ne[1] || x->ne[2] != norm->ne[2]) continue;
        if (norm->ne[1] != mul->ne[1]  || norm->ne[2] != mul->ne[2]) continue;
        if (mul->ne[1]  != add->ne[1]  || mul->ne[2] != add->ne[2])  continue;
        if (!is_1d_f32_affine(w, x->ne[0]))            continue;
        if (!is_1d_f32_affine(b, x->ne[0]))            continue;

        // Already-claimed nodes (a node can't be both an anchor and a
        // follower). Defensive — the chain shape above is locally unique.
        if (g_ln_anchors.count(add) || g_ln_followers.count(norm) ||
            g_ln_followers.count(mul)) {
            continue;
        }

        float eps = 0.0f;
        std::memcpy(&eps, norm->op_params, sizeof(float));
        if (!(eps >= 0.0f)) continue;

        LayerNormEntry e;
        e.x_node    = x;
        e.w_node    = w;
        e.b_node    = b;
        e.norm_node = norm;
        e.mul_node  = mul;
        e.H         = (int) x->ne[0];
        e.n_pos     = (int) flat_n_pos(x);
        e.eps       = eps;

        g_ln_anchors[add] = e;
        g_ln_followers.insert(norm);
        g_ln_followers.insert(mul);
        ++g_ln_groups_built;
    }
}

void build_qkv_prep_plan(const ggml_cgraph * cgraph) {
    g_qkv_copy_anchors.clear();
    g_qkv_split_anchors.clear();
    g_qkv_followers.clear();
    if (!cgraph || !g_qkv_enabled) return;

    const int n = ggml_graph_n_nodes((ggml_cgraph *) cgraph);

    for (int i = 0; i < n; ++i) {
        ggml_tensor * add = ggml_graph_node((ggml_cgraph *) cgraph, i);
        if (!add || add->op != GGML_OP_ADD) continue;

        const ggml_tensor * mm = add->src[0];
        if (!mm || mm->op != GGML_OP_MUL_MAT) continue;
        const ggml_tensor * w = mm->src[0];
        if (!w || !w->name[0]) continue;
        if (!std::strstr(w->name, "attn_qkv.weight")) continue;

        // qkv bias-add. ne=(3*H, n_pos[, n_batch]), F32 contiguous.
        // n_batch>1 is the encode_batch path (text). For 2D we keep n_batch=1.
        if (add->type != GGML_TYPE_F32) continue;
        if (add->ne[3] != 1) continue;
        if (add->nb[0] != ggml_type_size(add->type)) continue;
        if (add->nb[1] != add->nb[0] * add->ne[0]) continue;
        if (add->ne[2] > 1 && add->nb[2] != add->nb[1] * add->ne[1]) continue;

        const int64_t triH = add->ne[0];
        if (triH % 3 != 0) continue;
        const int H_block = (int) (triH / 3);
        const int n_pos   = (int) add->ne[1];
        const int n_batch = (int) add->ne[2];
        const size_t f32_size = sizeof(float);

        // Find the 3 view children of this add by walking forward.
        const ggml_tensor * v_q = nullptr, * v_k = nullptr, * v_v = nullptr;
        for (int j = i + 1; j < n; ++j) {
            ggml_tensor * c = ggml_graph_node((ggml_cgraph *) cgraph, j);
            if (!c || c->op != GGML_OP_VIEW) continue;
            if (c->src[0] != add) continue;
            const size_t off = c->view_offs;
            if (off == 0 && !v_q)                        v_q = c;
            else if (off == (size_t)(H_block * f32_size) && !v_k) v_k = c;
            else if (off == (size_t)(2 * H_block * f32_size) && !v_v) v_v = c;
            if (v_q && v_k && v_v) break;
        }
        if (!v_q || !v_k || !v_v) continue;

        // For each (Q, K, V) walk: view → permute → cont → pad → optional cpy(F16).
        // Returns true if the chain is well-formed (cont + pad reachable;
        // cast required for K/V, must NOT exist for Q).
        auto walk = [&](const ggml_tensor * v_node,
                        const ggml_tensor * & perm,
                        const ggml_tensor * & cont,
                        const ggml_tensor * & pad,
                        const ggml_tensor * & cast,
                        bool                  expect_cast) -> bool {
            perm = cont = pad = cast = nullptr;
            const ggml_tensor * src_for_perm = v_node;
            for (int j = i + 1; j < n; ++j) {
                ggml_tensor * c = ggml_graph_node((ggml_cgraph *) cgraph, j);
                if (!c) continue;
                if (!perm && c->op == GGML_OP_PERMUTE && c->src[0] == src_for_perm) perm = c;
                else if (!cont && c->op == GGML_OP_CONT && perm && c->src[0] == perm) cont = c;
                else if (!pad  && c->op == GGML_OP_PAD  && cont && c->src[0] == cont) pad  = c;
                else if (!cast && c->op == GGML_OP_CPY  && pad  && c->src[0] == pad &&
                         c->type == GGML_TYPE_F16) {
                    cast = c;
                    break;
                }
            }
            if (!perm || !cont || !pad) return false;
            if (expect_cast && !cast)   return false;
            if (!expect_cast && cast)   return false;
            return true;
        };

        const ggml_tensor * q_perm, * q_cont, * q_pad, * q_cast;
        const ggml_tensor * k_perm, * k_cont, * k_pad, * k_cast;
        const ggml_tensor * v_perm, * v_cont, * v_pad, * v_cast;
        if (!walk(v_q, q_perm, q_cont, q_pad, q_cast, /*expect_cast=*/false)) continue;
        if (!walk(v_k, k_perm, k_cont, k_pad, k_cast, /*expect_cast=*/true))  continue;
        if (!walk(v_v, v_perm, v_cont, v_pad, v_cast, /*expect_cast=*/true))  continue;

        // Shape sanity. The view's ne is (d_head, n_head, n_pos); the pad
        // result's ne[0] is d_pad (≥ d_head); n_pos and n_head must agree.
        const int d_head = (int) v_q->ne[0];
        const int n_head = (int) v_q->ne[1];
        const int d_pad  = (int) q_pad->ne[0];
        if (d_head <= 0 || n_head <= 0 || d_pad < d_head) continue;
        if (q_pad->ne[1] != n_pos || q_pad->ne[2] != n_head) continue;
        if (k_cast->ne[0] != d_pad || k_cast->ne[1] != n_pos || k_cast->ne[2] != n_head) continue;
        if (v_cast->ne[0] != d_pad || v_cast->ne[1] != n_pos || v_cast->ne[2] != n_head) continue;
        // The split-pad-cast destinations carry the prompt batch on ne[3].
        if (q_pad->ne[3]  != n_batch) continue;
        if (k_cast->ne[3] != n_batch) continue;
        if (v_cast->ne[3] != n_batch) continue;

        // Split anchor = whichever of {q_pad, k_cast, v_cast} appears
        // LAST in topo order. By that point all the upstream cont/pad/cast
        // followers have fired; the scratch buffer (written at qkv_add
        // anchor earlier in topo) is still alive.
        const ggml_tensor * split_anchor = nullptr;
        int split_anchor_idx = -1;
        for (int j = i + 1; j < n; ++j) {
            ggml_tensor * c = ggml_graph_node((ggml_cgraph *) cgraph, j);
            if (c == q_pad || c == k_cast || c == v_cast) {
                if (j > split_anchor_idx) { split_anchor_idx = j; split_anchor = c; }
            }
        }
        if (!split_anchor) continue;

        // Skip if any node is already claimed.
        if (g_qkv_copy_anchors.count(add) ||
            g_qkv_split_anchors.count(split_anchor) ||
            g_qkv_followers.count(q_cont) || g_qkv_followers.count(q_pad) ||
            g_qkv_followers.count(k_cont) || g_qkv_followers.count(k_pad) ||
            g_qkv_followers.count(k_cast) ||
            g_qkv_followers.count(v_cont) || g_qkv_followers.count(v_pad) ||
            g_qkv_followers.count(v_cast)) {
            continue;
        }

        QkvPrepEntry e;
        e.mm_node      = mm;
        e.bias_node    = add->src[1];   // qkv bias
        e.qkv_add      = add;
        e.q_pad_dst    = q_pad;
        e.k_cast_dst   = k_cast;
        e.v_cast_dst   = v_cast;
        e.split_anchor = split_anchor;
        e.H            = H_block;
        e.n_pos        = n_pos;
        e.n_head       = n_head;
        e.d_head       = d_head;
        e.d_pad        = d_pad;
        e.n_batch      = n_batch;
        g_qkv_copy_anchors[add]            = e;
        g_qkv_split_anchors[split_anchor]  = e;

        auto add_follower = [&](const ggml_tensor * t) {
            if (t && t != split_anchor) g_qkv_followers.insert(t);
        };
        add_follower(q_cont); add_follower(q_pad);
        add_follower(k_cont); add_follower(k_pad); add_follower(k_cast);
        add_follower(v_cont); add_follower(v_pad); add_follower(v_cast);
        ++g_qkv_groups_built;
    }
}

// Phase A2 plan-builder. Walks every ADD/UNARY node looking for two patterns
// rooted at a (mul_mat + 1D-bias) "inner" node:
//
//   P1: outer = ADD(inner_add, residual_2d), where inner_add = ADD(mm, bias_1d)
//   P2: outer = UNARY/GELU(inner_add),       where inner_add = ADD(mm, bias_1d)
//
// Anchor = outer; follower = inner_add. Outer's dst gets written directly by
// the fused kernel (reads mm.data + bias.data + residual.data, writes to
// outer.dst). The inner_add's compute is skipped — its slot may alias with
// mm.dst's slot (same shape, disjoint lifetimes), and at outer.idx the slot
// still holds mm's value because nothing else writes to it.
//
// Disjoint from A0/A1: A0 anchors on (NORM-prefixed) ADDs, A1 anchors on the
// QKV bias-add specifically (filtered by weight name "attn_qkv.weight"). Tail
// patterns require src[0].op == GGML_OP_ADD or live above a UNARY, neither of
// which the A0/A1 anchors match. Defensive `count(...)` checks below skip any
// node already claimed by a prior plan.
void build_tail_fusion_plan(const ggml_cgraph * cgraph) {
    g_br_anchors.clear();
    g_bg_anchors.clear();
    g_tail_followers.clear();
    if (!cgraph || !g_tail_enabled) return;

    // Permissive predicate (allows 3D-contig from encode_batch).
    auto is_2d_f32_contig = is_2d_or_3d_f32_contiguous;

    auto is_1d_f32_bias = [](const ggml_tensor * t, int64_t H) -> bool {
        if (!t) return false;
        if (t->type != GGML_TYPE_F32) return false;
        if (t->ne[0] != H) return false;
        if (t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1) return false;
        return true;
    };

    auto is_already_claimed = [](const ggml_tensor * node) -> bool {
        if (!node) return false;
        return g_ln_anchors.count(node)        || g_ln_followers.count(node) ||
               g_qkv_copy_anchors.count(node)  || g_qkv_split_anchors.count(node) ||
               g_qkv_followers.count(node)     ||
               g_br_anchors.count(node)        || g_bg_anchors.count(node) ||
               g_tail_followers.count(node);
    };

    auto match_inner = [&](const ggml_tensor * inner_add,
                           const ggml_tensor *& mm,
                           const ggml_tensor *& bias,
                           int & H, int & n_pos) -> bool {
        if (!inner_add || inner_add->op != GGML_OP_ADD) return false;
        const ggml_tensor * a = inner_add->src[0];
        const ggml_tensor * b = inner_add->src[1];
        if (!a || !b) return false;
        if (a->op != GGML_OP_MUL_MAT) return false;
        if (!is_2d_f32_contig(a))      return false;
        if (!is_2d_f32_contig(inner_add)) return false;
        if (a->ne[0] != inner_add->ne[0] ||
            a->ne[1] != inner_add->ne[1] ||
            a->ne[2] != inner_add->ne[2]) return false;
        H     = (int) a->ne[0];
        n_pos = (int) flat_n_pos(a);
        if (!is_1d_f32_bias(b, H)) return false;
        mm   = a;
        bias = b;
        return true;
    };

    // PRODUCTION GATE — fuse only patterns whose bias is a per-encoder-block
    // tensor (name contains "blk", i.e. "v.blk.{i}.*" or "t.blk.{i}.*"). The
    // probe head (`v.head.*`) and patch embedding (`v.patch_embd.*`) chains
    // empirically corrupt because gallocr's slot packing for those one-off,
    // mixed-shape sub-graphs doesn't keep mm.dst alive in the same physical
    // slot through to our late-anchor read. Per-block chains, by contrast,
    // are long and shape-uniform across 27 layers — gallocr packs them with
    // aggressive slot reuse that *does* preserve the value through our window.
    // Verified via per-pattern bisect: BLK_ONLY off → vision 0.65 cosine;
    // BLK_ONLY on → vision 0.9999. Same root family as the A1 gallocr trap.
    auto is_blk_bias = [](const ggml_tensor * bias) -> bool {
        if (!bias || !bias->name[0]) return false;
        return std::strstr(bias->name, "blk") != nullptr;
    };

    const int n = ggml_graph_n_nodes((ggml_cgraph *) cgraph);
    for (int i = 0; i < n; ++i) {
        ggml_tensor * outer = ggml_graph_node((ggml_cgraph *) cgraph, i);
        if (!outer) continue;

        // Defense-in-depth: also require inner_add to be the IMMEDIATELY
        // PRECEDING node in topo (outer at i, inner at i-1). Per-block
        // patterns always satisfy this; non-block ones often don't, so this
        // doubles-up on the BLK gate above.
        if (i < 1) continue;
        ggml_tensor * prev = ggml_graph_node((ggml_cgraph *) cgraph, i - 1);

        // Pattern P1: outer = ADD(inner_add, residual)
        if (outer->op == GGML_OP_ADD) {
            const ggml_tensor * inner    = outer->src[0];
            const ggml_tensor * residual = outer->src[1];
            const ggml_tensor * mm = nullptr, * bias = nullptr;
            int H = 0, n_pos = 0;
            if (!match_inner(inner, mm, bias, H, n_pos))   continue;
            if (inner != prev)                             continue;  // safety gate
            if (!is_blk_bias(bias))                        continue;
            if (!is_2d_f32_contig(residual))               continue;
            if (!is_2d_f32_contig(outer))                  continue;
            if (residual->ne[0] != H || (int) flat_n_pos(residual) != n_pos) continue;
            if (outer->ne[0]    != H || (int) flat_n_pos(outer)    != n_pos) continue;
            if (is_already_claimed(outer)) continue;
            if (is_already_claimed(inner)) continue;

            BiasResidualEntry e;
            e.mm_node       = mm;
            e.bias_node     = bias;
            e.residual_node = residual;
            e.inner_add     = inner;
            e.H             = H;
            e.n_pos         = n_pos;
            g_br_anchors[outer] = e;
            g_tail_followers.insert(inner);
            ++g_br_groups_built;
            continue;
        }

        // Pattern P2: outer = UNARY/GELU(inner_add)
        if (outer->op == GGML_OP_UNARY && ggml_get_unary_op(outer) == GGML_UNARY_OP_GELU) {
            const ggml_tensor * inner = outer->src[0];
            const ggml_tensor * mm = nullptr, * bias = nullptr;
            int H = 0, n_pos = 0;
            if (!match_inner(inner, mm, bias, H, n_pos)) continue;
            if (inner != prev)                            continue;  // safety gate
            if (!is_blk_bias(bias))                       continue;
            if (!is_2d_f32_contig(outer))                continue;
            if (outer->ne[0] != H || (int) flat_n_pos(outer) != n_pos) continue;
            if (is_already_claimed(outer)) continue;
            if (is_already_claimed(inner)) continue;

            BiasGeluEntry e;
            e.mm_node   = mm;
            e.bias_node = bias;
            e.inner_add = inner;
            e.H         = H;
            e.n_pos     = n_pos;
            g_bg_anchors[outer] = e;
            g_tail_followers.insert(inner);
            ++g_bg_groups_built;
            continue;
        }
    }
}

// Phase A3 plan-builder. Walks every CONT node looking for the post-FA pattern:
//
//   fa_out  = FLASH_ATTN_EXT(...)                       ne=(d_pad, n_head, n_pos)
//   view    = VIEW_3D(fa_out, ne=(d_head, n_head, n_pos), nb=fa_out->nb, off=0)
//   cont    = CONT(view)                                ne=(d_head, n_head, n_pos)
//
// The view must:
//   * Be a 3D non-contiguous view of fa_out (offset 0, ne[0]<d_pad, n_head and
//     n_pos preserved, strides inherited from fa_out).
// The cont's dst is the input to a downstream reshape_2d → o_proj mul_mat;
// nothing else reads from view, so anchoring on cont is safe.
//
// Defensive: skip if cont's dst already claimed by another fusion. The view
// is metadata (no compute) so it's neither anchor nor follower.
void build_post_fa_cont_plan(const ggml_cgraph * cgraph) {
    g_post_fa_anchors.clear();
    if (!cgraph || !g_post_fa_enabled) return;

    const int n = ggml_graph_n_nodes((ggml_cgraph *) cgraph);
    for (int i = 0; i < n; ++i) {
        ggml_tensor * cont = ggml_graph_node((ggml_cgraph *) cgraph, i);
        if (!cont || cont->op != GGML_OP_CONT) continue;
        if (cont->type != GGML_TYPE_F32)       continue;

        const ggml_tensor * view = cont->src[0];
        if (!view || view->op != GGML_OP_VIEW) continue;
        if (view->type != GGML_TYPE_F32)       continue;
        if (view->view_offs != 0)              continue;

        const ggml_tensor * fa_out = view->src[0];
        if (!fa_out || fa_out->op != GGML_OP_FLASH_ATTN_EXT) continue;

        // Must slice only d-axis (axis 0); other axes preserved. The optional
        // batch dim sits on ne[3] (encode_batch path); n_batch=1 in the unbatched
        // text/vision case.
        const int d_head = (int) view->ne[0];
        const int n_head = (int) view->ne[1];
        const int n_pos  = (int) view->ne[2];
        const int d_pad  = (int) fa_out->ne[0];
        const int n_batch = (int) view->ne[3];
        if (d_head <= 0 || d_head >= d_pad)         continue;
        if (n_head != (int) fa_out->ne[1])          continue;
        if (n_pos  != (int) fa_out->ne[2])          continue;
        if (n_batch != (int) fa_out->ne[3])         continue;

        // View must inherit fa_out's strides (we rely on the (d_pad, n_head,
        // n_pos[, n_batch]) physical layout).
        if (view->nb[1] != fa_out->nb[1])           continue;
        if (view->nb[2] != fa_out->nb[2])           continue;
        if (n_batch > 1 && view->nb[3] != fa_out->nb[3]) continue;

        // cont dst must be contiguous (d_head, n_head, n_pos[, n_batch]) F32.
        if (cont->ne[0] != d_head  || cont->ne[1] != n_head ||
            cont->ne[2] != n_pos   || cont->ne[3] != n_batch) continue;
        if (cont->nb[0] != ggml_type_size(GGML_TYPE_F32))                           continue;
        if (cont->nb[1] != cont->nb[0] * cont->ne[0])                               continue;
        if (cont->nb[2] != cont->nb[1] * cont->ne[1])                               continue;
        if (n_batch > 1 && cont->nb[3] != cont->nb[2] * cont->ne[2])                continue;

        // Skip if already claimed by another fusion (defensive — these patterns
        // shouldn't overlap with LN/QKV/tail anchors but cheap to guard).
        if (g_ln_anchors.count(cont)        || g_ln_followers.count(cont) ||
            g_qkv_copy_anchors.count(cont)  || g_qkv_split_anchors.count(cont) ||
            g_qkv_followers.count(cont)     ||
            g_br_anchors.count(cont)        || g_bg_anchors.count(cont) ||
            g_tail_followers.count(cont)    ||
            g_post_fa_anchors.count(cont)) {
            continue;
        }

        PostFaContEntry e;
        e.fa_out_node = fa_out;
        e.d_head      = d_head;
        e.d_pad       = d_pad;
        e.n_head      = n_head;
        e.n_pos       = n_pos;
        e.n_batch     = n_batch;
        g_post_fa_anchors[cont] = e;
        ++g_post_fa_groups_built;
    }
}

}  // namespace

// ----------------------------------------------------------------------------
// Hooks
// ----------------------------------------------------------------------------

extern "C" void siglip2_graph_begin_hook(
        ggml_backend_cuda_context * /*ctx*/,
        const ggml_cgraph *         cgraph) {
    if (!g_enabled) return;
    build_ln_plan(cgraph);
    build_qkv_prep_plan(cgraph);
    build_tail_fusion_plan(cgraph);
    build_post_fa_cont_plan(cgraph);

    if (g_log_budget > 0) {
        const int n = cgraph ? ggml_graph_n_nodes((ggml_cgraph *) cgraph) : 0;
        std::fprintf(stderr,
            "[siglip2-megakernel] graph_begin: LN groups=%zu (followers=%zu) "
            "QKV groups=%zu (copy_anchors=%zu split_anchors=%zu followers=%zu) "
            "tail bias-residual=%zu bias-gelu=%zu (followers=%zu) "
            "post-FA cont=%zu  total_nodes=%d\n",
            g_ln_anchors.size(),  g_ln_followers.size(),
            g_qkv_groups_built,
            g_qkv_copy_anchors.size(), g_qkv_split_anchors.size(),
            g_qkv_followers.size(),
            g_br_anchors.size(),  g_bg_anchors.size(),
            g_tail_followers.size(),
            g_post_fa_anchors.size(),
            n);
        --g_log_budget;
    }
}

extern "C" bool siglip2_op_hook(
        ggml_backend_cuda_context * /*ctx*/,
        ggml_tensor *               dst,
        cudaStream_t                stream) {
    if (!g_enabled || !dst) return false;

    // Fast path: follower no-op (LN's norm+mul, QKV-prep's cont/pad/cast, or
    // the inner bias-add of a tail fusion).
    if (g_ln_followers.count(dst)) {
        ++g_ln_follower_fires;
        return true;
    }
    if (g_qkv_followers.count(dst)) {
        ++g_qkv_follower_fires;
        return true;
    }
    if (g_tail_followers.count(dst)) {
        ++g_tail_follower_fires;
        return true;
    }

    // QKV-prep anchors. Two phases per block:
    //   (1) qkv_add  → copy mm + bias to scratch (replaces the bias-add launch)
    //   (2) v_cast   → split-permute-pad-cast from scratch to FA inputs
    if (g_qkv_enabled) {
        // Copy phase.
        auto it_c = g_qkv_copy_anchors.find(dst);
        if (it_c != g_qkv_copy_anchors.end()) {
            const QkvPrepEntry & e = it_c->second;
            const float * mm   = e.mm_node   ? (const float *) e.mm_node->data   : nullptr;
            const float * bias = e.bias_node ? (const float *) e.bias_node->data : nullptr;
            if (!mm || !bias) return false;

            const int triH = 3 * e.H;
            const size_t need_b = (size_t) triH * (size_t) e.n_pos
                                * (size_t) e.n_batch * sizeof(float);
            float * scratch = (float *) g_active_qkv_scratch_dptr;
            if (!scratch || g_active_qkv_scratch_cap < need_b) return false;

            profile_record_inline(kProfTagQkvCopy, stream);
            launch_qkv_copy_to_scratch(mm, bias, scratch,
                                       triH, e.n_pos, e.n_batch, stream);
            ++g_qkv_anchor_fires;
            return true;
        }
        // Split phase.
        auto it_s = g_qkv_split_anchors.find(dst);
        if (it_s != g_qkv_split_anchors.end()) {
            const QkvPrepEntry & e = it_s->second;
            float * q_pad  = e.q_pad_dst   ? (float *) e.q_pad_dst->data   : nullptr;
            void  * k_cast = e.k_cast_dst  ? e.k_cast_dst->data            : nullptr;
            void  * v_cast = e.v_cast_dst  ? e.v_cast_dst->data            : nullptr;
            float * scratch = (float *) g_active_qkv_scratch_dptr;
            if (!q_pad || !k_cast || !v_cast || !scratch) return false;

            profile_record_inline(kProfTagQkvSplit, stream);
            launch_fused_qkv_prep(scratch, q_pad, k_cast, v_cast,
                                  e.H, e.n_pos, e.n_head, e.d_head, e.d_pad,
                                  e.n_batch, stream);
            ++g_qkv_anchor_fires;
            return true;
        }
    }

    // Phase A3 — post-FA cont anchor. Anchored on the cont; reads the strided
    // FA output and writes contiguous (d_head, n_head, n_pos) to dst->data.
    // Same launch count as ggml's cont; specialized indexing only.
    if (g_post_fa_enabled) {
        auto it_pf = g_post_fa_anchors.find(dst);
        if (it_pf != g_post_fa_anchors.end()) {
            const PostFaContEntry & e = it_pf->second;
            const float * fa_out = e.fa_out_node ? (const float *) e.fa_out_node->data : nullptr;
            float       * y      = (float *) dst->data;
            if (!fa_out || !y) return false;
            profile_record_inline(kProfTagPostFA, stream);
            launch_fused_post_fa_cont(fa_out, y, e.d_head, e.d_pad, e.n_head, e.n_pos, e.n_batch, stream);
            ++g_post_fa_anchor_fires;
            return true;
        }
    }

    // Tail-fusion anchors. Outer op (residual_add or gelu) reads mm.data
    // directly — by gallocr's static analysis this slot may also host
    // bias_add.dst and outer.dst (same shape, disjoint lifetimes), so the
    // value persists from mm.idx through outer.idx because the inner bias_add
    // is a follower (no compute, no overwrite).
    if (g_tail_enabled) {
        auto it_br = g_br_anchors.find(dst);
        if (it_br != g_br_anchors.end()) {
            const BiasResidualEntry & e = it_br->second;
            const float * mm   = e.mm_node       ? (const float *) e.mm_node->data       : nullptr;
            const float * bias = e.bias_node     ? (const float *) e.bias_node->data     : nullptr;
            const float * res  = e.residual_node ? (const float *) e.residual_node->data : nullptr;
            float       * y    = (float *) dst->data;
            if (!mm || !bias || !res || !y) return false;
            profile_record_inline(kProfTagTailResid, stream);
            launch_fused_bias_residual(mm, bias, res, y, e.H, e.n_pos, stream);
            ++g_tail_anchor_fires;
            return true;
        }
        auto it_bg = g_bg_anchors.find(dst);
        if (it_bg != g_bg_anchors.end()) {
            const BiasGeluEntry & e = it_bg->second;
            const float * mm   = e.mm_node   ? (const float *) e.mm_node->data   : nullptr;
            const float * bias = e.bias_node ? (const float *) e.bias_node->data : nullptr;
            float       * y    = (float *) dst->data;
            if (!mm || !bias || !y) return false;
            profile_record_inline(kProfTagTailGelu, stream);
            launch_fused_bias_gelu(mm, bias, y, e.H, e.n_pos, stream);
            ++g_tail_anchor_fires;
            return true;
        }
    }

    // LayerNorm anchor.
    auto it = g_ln_anchors.find(dst);
    if (it == g_ln_anchors.end()) return false;

    const LayerNormEntry & e = it->second;
    const float * x = e.x_node ? (const float *) e.x_node->data : nullptr;
    const float * w = e.w_node ? (const float *) e.w_node->data : nullptr;
    const float * b = e.b_node ? (const float *) e.b_node->data : nullptr;
    float       * y = (float *) dst->data;
    if (!x || !w || !b || !y) return false;
    profile_record_inline(kProfTagLN, stream);

    launch_fused_layernorm_affine(x, w, b, y, e.H, e.n_pos, e.eps, stream);
    ++g_ln_anchor_fires;
    return true;
}

// ----------------------------------------------------------------------------
// install() / is_installed()
// ----------------------------------------------------------------------------

namespace {
bool s_installed = false;
}

bool is_installed() { return s_installed; }

void set_active_qkv_scratch(void * dptr, std::size_t cap_bytes) {
    g_active_qkv_scratch_dptr = dptr;
    g_active_qkv_scratch_cap  = cap_bytes;
}

void clear_active_qkv_scratch() {
    g_active_qkv_scratch_dptr = nullptr;
    g_active_qkv_scratch_cap  = 0;
}

bool profile_enabled() { return profile_enabled_check(); }

void log_device_stream_priority_range() {
    int least = 0, greatest = 0;
    if (cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess) {
        std::fprintf(stderr, "[siglip2-megakernel] stream priority range: query failed\n");
        return;
    }
    // CUDA convention: LOWER int = HIGHER priority. `least` = least priority,
    // `greatest` = greatest priority (most negative).
    std::fprintf(stderr,
        "[siglip2-megakernel] stream priority range: [%d, %d] (greatest..least; %s)\n",
        greatest, least,
        greatest == least
            ? "single-priority device — HIGH/LOW hints are no-ops"
            : "multi-priority — HIGH wins SM contention");
}

void profile_after_encode(const char * label, void * /*stream_v*/) {
    if (!profile_enabled_check()) return;
    // Even on a graph-replay (where op_hook didn't fire and so used==0 this
    // call), the events we captured into the graph DID fire and got fresh
    // timestamps. So we don't gate on used==0 — instead we use the SIZE of
    // the events vector as the total count to walk. This way every replay
    // gets a fresh dump.
    const size_t n = g_profile.events.size();
    if (n < 2) return;

    cudaEventSynchronize(g_profile.events[n - 1]);

    struct Bucket { double sum_ms = 0; int count = 0; };
    Bucket buckets[kProfNumTags];
    double total_ms = 0;
    int    fail_count = 0;
    cudaError_t last_err = cudaSuccess;
    for (size_t i = 0; i + 1 < n; ++i) {
        float dt = 0;
        cudaError_t e = cudaEventElapsedTime(&dt, g_profile.events[i], g_profile.events[i+1]);
        if (e != cudaSuccess) {
            ++fail_count;
            last_err = e;
            continue;
        }
        const int t = g_profile.tags[i];
        if (t >= 0 && t < kProfNumTags) {
            buckets[t].sum_ms += dt;
            buckets[t].count++;
            total_ms += dt;
        }
    }

    static const char * tag_name[kProfNumTags] = {
        "LN          ", "QKV-copy    ", "QKV-split   ", "post-FA cont",
        "tail-bias+rs", "tail-bias+gl", "<end>       ",
    };
    std::fprintf(stderr,
        "[profile %s] anchored=%.3f ms across %zu segments (failed=%d, last_err=%s)\n",
        label, total_ms, n - 1, fail_count,
        fail_count > 0 ? cudaGetErrorString(last_err) : "ok");
    for (int t = 0; t < kProfNumTags - 1; ++t) {
        if (buckets[t].count == 0) continue;
        std::fprintf(stderr,
            "  %s  sum=%.3f ms  fires=%4d  avg=%.4f ms/fire  share=%.1f%%\n",
            tag_name[t], buckets[t].sum_ms, buckets[t].count,
            buckets[t].sum_ms / buckets[t].count,
            total_ms > 0 ? 100.0 * buckets[t].sum_ms / total_ms : 0.0);
    }

    // Reset used so the NEXT encode (whether warmup or replay) starts at
    // index 0. Events themselves stay alive: warmup-1/warmup-2 calls into
    // op_hook will re-record them in place; replays bypass op_hook but the
    // captured graph re-fires the same event objects, refreshing timestamps.
    g_profile.used = 0;
}

bool install() {
    if (s_installed) return false;

    if (const char * d = std::getenv("SIGLIP2_DISABLE_MEGAKERNEL")) {
        if (d[0] != '\0' && std::strcmp(d, "0") != 0) {
            return false;
        }
    }

    if (const char * v = std::getenv("SIGLIP2_MEGAKERNEL_VERBOSE")) {
        if (v[0] != '\0' && std::strcmp(v, "0") != 0) g_log_budget = 5;
    }

    if (const char * d = std::getenv("SIGLIP2_DISABLE_QKV_PREP")) {
        if (d[0] != '\0' && std::strcmp(d, "0") != 0) g_qkv_enabled = false;
    }

    if (const char * d = std::getenv("SIGLIP2_DISABLE_TAIL_FUSION")) {
        if (d[0] != '\0' && std::strcmp(d, "0") != 0) g_tail_enabled = false;
    }

    if (const char * d = std::getenv("SIGLIP2_DISABLE_POST_FA_CONT")) {
        if (d[0] != '\0' && std::strcmp(d, "0") != 0) g_post_fa_enabled = false;
    }

    ggml_cuda_set_graph_begin_hook(siglip2_graph_begin_hook);
    ggml_cuda_set_op_hook(siglip2_op_hook);
    s_installed = true;

    std::fprintf(stderr,
        "[siglip2-megakernel] installed: fused-LN%s%s%s\n",
        g_qkv_enabled     ? " + QKV-prep (copy→scratch + split-permute-pad-cast, 9 ops → 2)" : "",
        g_tail_enabled    ? " + tail (bias+residual, bias+gelu, 3 ops/block → 0 anchor launches)" : "",
        g_post_fa_enabled ? " + post-FA cont (specialized strided slice→contig)" : "");
    return true;
}

}  // namespace siglip2_megakernel
