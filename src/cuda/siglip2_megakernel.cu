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
#include <unordered_map>
#include <unordered_set>

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
// Fused QKV-prep kernel
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

    const int dst_idx = h * d_pad * n_pos + p * d_pad + d;

    if (d < d_head) {
        const int src_idx = p * 3 * H + c * H + h * d_head + d;
        const float v = qkv[src_idx];
        if (c == 0) {
            q_pad[dst_idx] = v;
        } else if (c == 1) {
            k_cast[dst_idx] = __float2half(v);
        } else {
            v_cast[dst_idx] = __float2half(v);
        }
    } else {
        // Pad slot — exact zero (no denormals).
        if (c == 0) {
            q_pad[dst_idx] = 0.0f;
        } else if (c == 1) {
            k_cast[dst_idx] = __ushort_as_half((unsigned short) 0);
        } else {
            v_cast[dst_idx] = __ushort_as_half((unsigned short) 0);
        }
    }
}

}  // namespace

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
        cudaStream_t  stream) {
    const dim3 grid((unsigned) n_pos, (unsigned) (3 * n_head), 1);
    const dim3 block((unsigned) d_pad, 1, 1);
    fused_qkv_prep_kernel<<<grid, block, 0, stream>>>(
        qkv, q_pad,
        static_cast<__half *>(k_cast),
        static_cast<__half *>(v_cast),
        H, n_head, d_head, d_pad, n_pos);
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

struct QkvPrepEntry {
    const ggml_tensor * qkv_node    = nullptr;  // qkv bias-add (data source)
    const ggml_tensor * q_pad_dst   = nullptr;  // F32 dst we write to
    const ggml_tensor * k_cast_dst  = nullptr;  // F16 dst we write to
    const ggml_tensor * v_cast_dst  = nullptr;  // F16 dst we write to
    int H      = 0;
    int n_pos  = 0;
    int n_head = 0;
    int d_head = 0;
    int d_pad  = 0;
};

namespace {

std::unordered_map<const ggml_tensor *, LayerNormEntry> g_ln_anchors;
std::unordered_set<const ggml_tensor *>                 g_ln_followers;

std::unordered_map<const ggml_tensor *, QkvPrepEntry>   g_qkv_anchors;
std::unordered_set<const ggml_tensor *>                 g_qkv_followers;

bool     g_enabled              = true;
// Phase A1 QKV-prep is PARKED default-off: the late-anchor design (firing at
// V_cast.idx so all output dsts are reachable) means by the time the fused
// kernel reads `qkv_add->data`, ggml's gallocr has freed that slot (its
// static analysis says qkv_add's last consumer is the conts, which we made
// followers — gallocr doesn't know we deferred). The slot may have been
// reused by a different tensor mid-graph. For vision (n_pos=729) the
// aliasing happens to land on something benign and the output is bit-clean
// vs A1-OFF; for text (n_pos=64) the aliasing corrupts a live tensor and
// every encode produces NaN. The fix is a 2-kernel design:
//   1. Anchor qkv_add: copy mm + bias into a persistent device-side scratch
//      buffer (sized for the hottest 3*H*n_pos shape; ~10 MiB).
//   2. Anchor V_cast: split-permute-pad-cast from scratch → Q_pad/K_cast/V_cast.
// Saves 7 launches per block (= 9 fused → 2 kernels). Set
// SIGLIP2_ENABLE_QKV_PREP=1 to opt in for development; do NOT ship default-on
// until the scratch-buffer redesign lands. See HANDOFF-megakernel-v0.md
// "Phase A1 — gallocr aliasing trap" for receipts.
bool     g_qkv_enabled          = false;
uint64_t g_ln_groups_built      = 0;
uint64_t g_ln_anchor_fires      = 0;
uint64_t g_ln_follower_fires    = 0;
uint64_t g_qkv_groups_built     = 0;
uint64_t g_qkv_anchor_fires     = 0;
uint64_t g_qkv_follower_fires   = 0;
int      g_log_budget           = 0;

bool is_2d_f32_contiguous(const ggml_tensor * t) {
    if (!t) return false;
    if (t->type != GGML_TYPE_F32) return false;
    if (t->ne[2] != 1 || t->ne[3] != 1) return false;
    if (t->nb[0] != ggml_type_size(t->type)) return false;
    if (t->nb[1] != t->nb[0] * t->ne[0])     return false;
    return true;
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

        if (!is_2d_f32_contiguous(x))                  continue;
        if (!is_2d_f32_contiguous(norm))               continue;
        if (!is_2d_f32_contiguous(mul))                continue;
        if (!is_2d_f32_contiguous(add))                continue;
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
        e.n_pos     = (int) x->ne[1];
        e.eps       = eps;

        g_ln_anchors[add] = e;
        g_ln_followers.insert(norm);
        g_ln_followers.insert(mul);
        ++g_ln_groups_built;
    }
}

void build_qkv_prep_plan(const ggml_cgraph * cgraph) {
    g_qkv_anchors.clear();
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

        // qkv bias-add. ne=(3*H, n_pos), F32 contiguous.
        if (add->type != GGML_TYPE_F32) continue;
        if (add->ne[2] != 1 || add->ne[3] != 1) continue;
        if (add->nb[0] != ggml_type_size(add->type)) continue;

        const int64_t triH = add->ne[0];
        if (triH % 3 != 0) continue;
        const int H_block = (int) (triH / 3);
        const int n_pos   = (int) add->ne[1];
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

        // Anchor = whichever of {q_pad, k_cast, v_cast} appears LAST in
        // the cgraph topo order. By that point ggml has dispatched (and
        // we've followered) every other node in the chain, and the qkv
        // add's data is settled.
        const ggml_tensor * anchor = nullptr;
        int anchor_idx = -1;
        for (int j = i + 1; j < n; ++j) {
            ggml_tensor * c = ggml_graph_node((ggml_cgraph *) cgraph, j);
            if (c == q_pad || c == k_cast || c == v_cast) {
                if (j > anchor_idx) { anchor_idx = j; anchor = c; }
            }
        }
        if (!anchor) continue;

        // Skip if any of these nodes are already claimed by a different
        // chain (defensive — locally unique by weight name + offset).
        if (g_qkv_anchors.count(anchor) ||
            g_qkv_followers.count(q_cont) || g_qkv_followers.count(q_pad) ||
            g_qkv_followers.count(k_cont) || g_qkv_followers.count(k_pad) ||
            g_qkv_followers.count(k_cast) ||
            g_qkv_followers.count(v_cont) || g_qkv_followers.count(v_pad) ||
            g_qkv_followers.count(v_cast)) {
            continue;
        }

        QkvPrepEntry e;
        e.qkv_node   = add;
        e.q_pad_dst  = q_pad;
        e.k_cast_dst = k_cast;
        e.v_cast_dst = v_cast;
        e.H          = H_block;
        e.n_pos      = n_pos;
        e.n_head     = n_head;
        e.d_head     = d_head;
        e.d_pad      = d_pad;
        g_qkv_anchors[anchor] = e;

        auto add_follower = [&](const ggml_tensor * t) {
            if (t && t != anchor) g_qkv_followers.insert(t);
        };
        add_follower(q_cont); add_follower(q_pad);
        add_follower(k_cont); add_follower(k_pad); add_follower(k_cast);
        add_follower(v_cont); add_follower(v_pad); add_follower(v_cast);
        ++g_qkv_groups_built;
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

    if (g_log_budget > 0) {
        const int n = cgraph ? ggml_graph_n_nodes((ggml_cgraph *) cgraph) : 0;
        std::fprintf(stderr,
            "[siglip2-megakernel] graph_begin: LN groups=%zu (followers=%zu) "
            "QKV groups=%zu (followers=%zu) total_nodes=%d\n",
            g_ln_anchors.size(),  g_ln_followers.size(),
            g_qkv_anchors.size(), g_qkv_followers.size(), n);
        --g_log_budget;
    }
}

extern "C" bool siglip2_op_hook(
        ggml_backend_cuda_context * /*ctx*/,
        ggml_tensor *               dst,
        cudaStream_t                stream) {
    if (!g_enabled || !dst) return false;

    // Fast path: follower no-op (LN's norm+mul, or QKV-prep's cont/pad/cast).
    if (g_ln_followers.count(dst)) {
        ++g_ln_follower_fires;
        return true;
    }
    if (g_qkv_followers.count(dst)) {
        ++g_qkv_follower_fires;
        return true;
    }

    // QKV-prep anchor: fire fused split-permute-pad-cast kernel.
    if (g_qkv_enabled) {
        auto it = g_qkv_anchors.find(dst);
        if (it != g_qkv_anchors.end()) {
            const QkvPrepEntry & e = it->second;
            const float * qkv = e.qkv_node ? (const float *) e.qkv_node->data : nullptr;
            float * q_pad     = e.q_pad_dst   ? (float *) e.q_pad_dst->data   : nullptr;
            void  * k_cast    = e.k_cast_dst  ? e.k_cast_dst->data            : nullptr;
            void  * v_cast    = e.v_cast_dst  ? e.v_cast_dst->data            : nullptr;
            if (!qkv || !q_pad || !k_cast || !v_cast) return false;

            launch_fused_qkv_prep(qkv, q_pad, k_cast, v_cast,
                                  e.H, e.n_pos, e.n_head, e.d_head, e.d_pad,
                                  stream);
            ++g_qkv_anchor_fires;
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

    if (const char * e = std::getenv("SIGLIP2_ENABLE_QKV_PREP")) {
        if (e[0] != '\0' && std::strcmp(e, "0") != 0) g_qkv_enabled = true;
    }

    ggml_cuda_set_graph_begin_hook(siglip2_graph_begin_hook);
    ggml_cuda_set_op_hook(siglip2_op_hook);
    s_installed = true;

    std::fprintf(stderr,
        "[siglip2-megakernel] installed: fused-LN%s\n",
        g_qkv_enabled ? " + QKV-prep (DEV-ONLY, broken on text — see HANDOFF-megakernel-v0.md)" : "");
    return true;
}

}  // namespace siglip2_megakernel
