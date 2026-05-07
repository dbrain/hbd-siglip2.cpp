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

namespace {

std::unordered_map<const ggml_tensor *, LayerNormEntry> g_ln_anchors;
std::unordered_set<const ggml_tensor *>                 g_ln_followers;

bool     g_enabled         = true;
uint64_t g_groups_built    = 0;
uint64_t g_anchor_fires    = 0;
uint64_t g_follower_fires  = 0;
int      g_log_budget      = 0;

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
        ++g_groups_built;
    }

    if (g_log_budget > 0) {
        std::fprintf(stderr,
            "[siglip2-megakernel] graph_begin: built %zu LN groups (followers=%zu, total nodes=%d)\n",
            g_ln_anchors.size(), g_ln_followers.size(), n);
        --g_log_budget;
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
}

extern "C" bool siglip2_op_hook(
        ggml_backend_cuda_context * /*ctx*/,
        ggml_tensor *               dst,
        cudaStream_t                stream) {
    if (!g_enabled || !dst) return false;

    // Fast path: follower no-op. Norm and mul nodes whose work has been
    // absorbed into the anchor's fused launch.
    if (g_ln_followers.count(dst)) {
        ++g_follower_fires;
        return true;
    }

    // Anchor: fire fused kernel writing into dst->data (the ADD's allocated
    // output). Resolve all source data lazily — ggml's allocator fills ->data
    // during dispatch, not at graph-begin.
    auto it = g_ln_anchors.find(dst);
    if (it == g_ln_anchors.end()) return false;

    const LayerNormEntry & e = it->second;
    const float * x = e.x_node ? (const float *) e.x_node->data : nullptr;
    const float * w = e.w_node ? (const float *) e.w_node->data : nullptr;
    const float * b = e.b_node ? (const float *) e.b_node->data : nullptr;
    float       * y = (float *) dst->data;
    if (!x || !w || !b || !y) return false;

    launch_fused_layernorm_affine(x, w, b, y, e.H, e.n_pos, e.eps, stream);
    ++g_anchor_fires;
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

    ggml_cuda_set_graph_begin_hook(siglip2_graph_begin_hook);
    ggml_cuda_set_op_hook(siglip2_op_hook);
    s_installed = true;

    std::fprintf(stderr,
        "[siglip2-megakernel] installed: fused-LN (norm+mul-w+add-b)\n");
    return true;
}

}  // namespace siglip2_megakernel
