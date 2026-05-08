// siglip2 custom MMQ — Phase C kernel-lap entry point.
//
// Stable baseline (post-v5): hand-rolled mma m16n8k32 INT8 kernel for
// siglip2's exact (Q8_0 W) × (F32 x) → F32 dst shape set. Cosine = 1.0
// (bit-clean) across all 9 Q8_0 microbench shapes; ratio cust/ggml on a
// clean GPU sits between 1.27× (low-M shapes that ggml under-utilizes) and
// 2.20× (high-M shapes where ggml's stream-K shines). cfe shape (attn_qkv
// M=320, K=1152, N=3456) measures ~1.74×.
//
// Iteration history (kept here so a future reader sees what was tried):
//   v1: 1 mma/warp/K-block, no fan-out, no scale cache.   2.3-7.1× slower.
//   v2: per-warp (N_SUBTILES × M_SUBTILES) fan-out.       2.2-2.9× slower.
//   v3: bumped block tile (128, 128).                     no win over v2.
//   v4: K_BATCH=4 shmem batching (no cp.async).           regressed (occupancy hit).
//   v5: scale cache in registers; load_generic for B.    1.27-2.20× (this file).
//   v5b: 4 warps × 1 (vs 8×1) — same perf as v5.          (this file's launcher).
//   v6: cp.async + double-buffered shmem.                 cosine drift on edge tiles;
//                                                         preserved on dbrain/phase-c-cpasync-wip.
//
// The remaining gap to ggml is dominated by stream-K work distribution
// (ggml runs nsm=28 fixed blocks each handling ~104 K-blocks of work; ours
// runs hundreds of small blocks). cp.async pipelining (v6 branch) is a
// secondary lever — needed but not sufficient on its own.
//
// Convention (matches the microbench / ggml_mul_mat semantics):
//   W:   (K, N)   Q8_0   K innermost; stored as N rows × K_BLOCKS of block_q8_0
//   x:   (K, M)   F32    K innermost; quantized on the fly to (act_qs, act_d)
//   dst: (N, M)   F32    N innermost; dst[m * N + n]  is the (n, m) entry
//
// Activation packing (see siglip2_custom_mmq.cuh):
//   act_qs : int8[M * K]   row-major in K within m, m-major in outer loop
//   act_d  : float[M * K_BLOCKS]   per-32K-block scale = max|x|/127

#include "siglip2_custom_mmq.cuh"

#include "ggml-cuda/mma.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"  // get_int_b2: 2-byte-aligned int read for block_q8_0::qs

#include <cuda_fp16.h>
#include <cstdio>

namespace siglip2_custom_mmq {

using namespace ggml_cuda_mma;

// ----------------------------------------------------------------------------
// Activation quantize-x: F32 (K, M) → packed (act_qs, act_d).
// ----------------------------------------------------------------------------

__global__ void quantize_activation_kernel(
        const float * __restrict__ x,
        int8_t      * __restrict__ act_qs,
        float       * __restrict__ act_d,
        int                        M)
{
    const int m       = blockIdx.x;
    const int k_block = blockIdx.y;
    const int lane    = threadIdx.x;

    if (m >= M) return;

    const float xv = x[m * K_FIXED + k_block * K_BLOCK_SIZE + lane];

    float a = fabsf(xv);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, off, 32));
    }

    const float d     = a / 127.f;
    const float inv_d = (a == 0.f) ? 0.f : (127.f / a);

    int q = __float2int_rn(xv * inv_d);
    if (q >  127) q =  127;
    if (q < -128) q = -128;

    act_qs[m * K_FIXED + k_block * K_BLOCK_SIZE + lane] = (int8_t) q;
    if (lane == 0) {
        act_d[m * K_BLOCKS + k_block] = d;
    }
}

void launch_quantize_activation(
        const float *         x,
        void *                scratch,
        int                   M,
        siglip2_cuda_stream_t s)
{
    cudaStream_t stream = (cudaStream_t) s;
    int8_t * qs = (int8_t *) scratch;
    float  * d  = (float  *) ((char *) scratch + (size_t) M * K_FIXED);

    dim3 grid(M, K_BLOCKS, 1);
    dim3 block(32, 1, 1);
    quantize_activation_kernel<<<grid, block, 0, stream>>>(x, qs, d, M);
}

// ----------------------------------------------------------------------------
// Custom MMQ kernel — v5/v5b.
// ----------------------------------------------------------------------------

template <int N_BLOCK, int M_BLOCK, int WARPS_N_, int WARPS_M_>
struct mmq_config {
    static constexpr int WARPS_N    = WARPS_N_;
    static constexpr int WARPS_M    = WARPS_M_;
    static constexpr int NWARPS     = WARPS_N * WARPS_M;
    static constexpr int THREADS    = NWARPS * 32;
    static constexpr int N_SUBTILES = N_BLOCK / (WARPS_N * 16);
    static constexpr int M_SUBTILES = M_BLOCK / (WARPS_M * 8);

    static_assert(N_BLOCK % (WARPS_N * 16) == 0, "bad N tiling");
    static_assert(M_BLOCK % (WARPS_M * 8)  == 0, "bad M tiling");

    static constexpr int W_INTS = N_BLOCK * (K_BLOCK_SIZE / 4);
    static constexpr int X_INTS = M_BLOCK * (K_BLOCK_SIZE / 4);
};

template <int N_BLOCK, int M_BLOCK, int WARPS_N_, int WARPS_M_, bool PROFILE = false>
__launch_bounds__(mmq_config<N_BLOCK, M_BLOCK, WARPS_N_, WARPS_M_>::THREADS, 1)
__global__ void custom_mmq_kernel(
        const block_q8_0 * __restrict__ w,
        const int8_t     * __restrict__ act_qs,
        const float      * __restrict__ act_d,
        float            * __restrict__ dst,
        int M, int N,
        ProfileCycles    * __restrict__ out_cycles = nullptr)
{
    using Cfg = mmq_config<N_BLOCK, M_BLOCK, WARPS_N_, WARPS_M_>;

    constexpr int WARPS_N    = Cfg::WARPS_N;
    constexpr int N_SUBTILES = Cfg::N_SUBTILES;
    constexpr int M_SUBTILES = Cfg::M_SUBTILES;

    const int warp_id  = threadIdx.y;
    const int lane     = threadIdx.x;
    const int warp_n   = warp_id % WARPS_N;
    const int warp_m   = warp_id / WARPS_N;

    const int block_n0 = blockIdx.x * N_BLOCK;
    const int block_m0 = blockIdx.y * M_BLOCK;

    // Per-thread scale endpoints (matches mma m16n8k32 register layout):
    //   l=0: (lane/4,     2*(lane%4))
    //   l=1: (lane/4,     2*(lane%4)+1)
    //   l=2: (lane/4 + 8, 2*(lane%4))
    //   l=3: (lane/4 + 8, 2*(lane%4)+1)
    const int n_in_lo = lane / 4;
    const int n_in_hi = n_in_lo + 8;
    const int m_in_lo = 2 * (lane % 4);
    const int m_in_hi = m_in_lo + 1;

    float Cf[N_SUBTILES][M_SUBTILES][4];
#pragma unroll
    for (int sn = 0; sn < N_SUBTILES; ++sn)
#pragma unroll
        for (int sm = 0; sm < M_SUBTILES; ++sm)
#pragma unroll
            for (int l = 0; l < 4; ++l)
                Cf[sn][sm][l] = 0.f;

    extern __shared__ int sm_data[];
    int   * smem_w  = sm_data;
    int   * smem_x  = smem_w + Cfg::W_INTS;
    float * smem_dw = (float *) (smem_x + Cfg::X_INTS);
    float * smem_dx = smem_dw + N_BLOCK;

    const int tid           = warp_id * 32 + lane;
    const int total_threads = Cfg::THREADS;

    // Profile: per-region cycle counters (only when PROFILE=true).
    unsigned long long cyc_load = 0, cyc_sync1 = 0, cyc_compute = 0, cyc_sync2 = 0;

    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        const unsigned long long t_load_start = PROFILE ? clock64() : 0ull;
        // -- Cooperative load: W slab + W scales + x slab + x scales.
#pragma unroll
        for (int i = tid; i < Cfg::W_INTS; i += total_threads) {
            const int n_row    = i / 8;
            const int j        = i % 8;
            const int n_global = block_n0 + n_row;
            int v = 0;
            if (n_global < N) {
                const block_q8_0 * blk = w + n_global * K_BLOCKS + kb;
                v = get_int_b2(blk->qs, j);
            }
            smem_w[i] = v;
        }
#pragma unroll
        for (int i = tid; i < N_BLOCK; i += total_threads) {
            const int n_global = block_n0 + i;
            float dw = 0.f;
            if (n_global < N) {
                const block_q8_0 * blk = w + n_global * K_BLOCKS + kb;
                dw = __half2float(blk->d);
            }
            smem_dw[i] = dw;
        }
#pragma unroll
        for (int i = tid; i < Cfg::X_INTS; i += total_threads) {
            const int m_row    = i / 8;
            const int j        = i % 8;
            const int m_global = block_m0 + m_row;
            int v = 0;
            if (m_global < M) {
                const int8_t * row = act_qs + m_global * K_FIXED + kb * K_BLOCK_SIZE;
                v = ((const int *) row)[j];
            }
            smem_x[i] = v;
        }
#pragma unroll
        for (int i = tid; i < M_BLOCK; i += total_threads) {
            const int m_global = block_m0 + i;
            smem_dx[i] = (m_global < M) ? act_d[m_global * K_BLOCKS + kb] : 0.f;
        }

        const unsigned long long t_sync1_start = PROFILE ? clock64() : 0ull;
        __syncthreads();
        const unsigned long long t_compute_start = PROFILE ? clock64() : 0ull;

        // -- Pre-cache per-thread scales for this warp's N strips and M strips.
        float dw_lo[N_SUBTILES], dw_hi[N_SUBTILES];
#pragma unroll
        for (int sn = 0; sn < N_SUBTILES; ++sn) {
            const int n_strip = warp_n * N_SUBTILES + sn;
            dw_lo[sn] = smem_dw[n_strip * 16 + n_in_lo];
            dw_hi[sn] = smem_dw[n_strip * 16 + n_in_hi];
        }
        float dx_lo[M_SUBTILES], dx_hi[M_SUBTILES];
#pragma unroll
        for (int sm = 0; sm < M_SUBTILES; ++sm) {
            const int m_strip = warp_m * M_SUBTILES + sm;
            dx_lo[sm] = smem_dx[m_strip * 8 + m_in_lo];
            dx_hi[sm] = smem_dx[m_strip * 8 + m_in_hi];
        }

        // -- Load A tiles for this K-block (one per N-subtile).
        tile<16, 8, int> A[N_SUBTILES];
#pragma unroll
        for (int sn = 0; sn < N_SUBTILES; ++sn) {
            const int n_strip = warp_n * N_SUBTILES + sn;
            const int * A_smem = smem_w + n_strip * 16 * (K_BLOCK_SIZE / 4);
            load_ldmatrix(A[sn], A_smem, K_BLOCK_SIZE / 4);
        }

        // -- For each M-subtile: load B + mma fan-out across N-subtiles.
#pragma unroll
        for (int sm = 0; sm < M_SUBTILES; ++sm) {
            const int m_strip = warp_m * M_SUBTILES + sm;
            const int * B_smem = smem_x + m_strip * 8 * (K_BLOCK_SIZE / 4);
            tile<8, 8, int> B;
            // load_generic: per-element shmem read; ggml mma path notes
            // "faster than load_ldmatrix" for tile<8,8,int> on Turing+ Ampere.
            load_generic(B, B_smem, K_BLOCK_SIZE / 4);

#pragma unroll
            for (int sn = 0; sn < N_SUBTILES; ++sn) {
                tile<16, 8, int> Cint;
#pragma unroll
                for (int l = 0; l < Cint.ne; ++l) Cint.x[l] = 0;
                mma(Cint, A[sn], B);

                Cf[sn][sm][0] += (float) Cint.x[0] * dw_lo[sn] * dx_lo[sm];
                Cf[sn][sm][1] += (float) Cint.x[1] * dw_lo[sn] * dx_hi[sm];
                Cf[sn][sm][2] += (float) Cint.x[2] * dw_hi[sn] * dx_lo[sm];
                Cf[sn][sm][3] += (float) Cint.x[3] * dw_hi[sn] * dx_hi[sm];
            }
        }

        const unsigned long long t_sync2_start = PROFILE ? clock64() : 0ull;
        __syncthreads();
        const unsigned long long t_iter_end = PROFILE ? clock64() : 0ull;

        if (PROFILE) {
            cyc_load    += (t_sync1_start  - t_load_start);
            cyc_sync1   += (t_compute_start - t_sync1_start);
            cyc_compute += (t_sync2_start  - t_compute_start);
            cyc_sync2   += (t_iter_end     - t_sync2_start);
        }
    }

    // -- Write back.
#pragma unroll
    for (int sn = 0; sn < N_SUBTILES; ++sn) {
        const int n_strip = warp_n * N_SUBTILES + sn;
#pragma unroll
        for (int sm = 0; sm < M_SUBTILES; ++sm) {
            const int m_strip = warp_m * M_SUBTILES + sm;
            const int n_base  = block_n0 + n_strip * 16;
            const int m_base  = block_m0 + m_strip * 8;
            const int n0 = n_base + n_in_lo;
            const int n1 = n_base + n_in_hi;
            const int m0 = m_base + m_in_lo;
            const int m1 = m_base + m_in_hi;
            if (n0 < N && m0 < M) dst[m0 * N + n0] = Cf[sn][sm][0];
            if (n0 < N && m1 < M) dst[m1 * N + n0] = Cf[sn][sm][1];
            if (n1 < N && m0 < M) dst[m0 * N + n1] = Cf[sn][sm][2];
            if (n1 < N && m1 < M) dst[m1 * N + n1] = Cf[sn][sm][3];
        }
    }

    // Profile: lane 0 of warp 0 of each block reports the block's totals.
    // Cycle counts are SM-clock cycles; all threads in a block share the same
    // SM clock so the deltas are coherent.
    if (PROFILE && warp_id == 0 && lane == 0) {
        const int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
        out_cycles[block_idx].load    = cyc_load;
        out_cycles[block_idx].sync1   = cyc_sync1;
        out_cycles[block_idx].compute = cyc_compute;
        out_cycles[block_idx].sync2   = cyc_sync2;
    }
}

// ----------------------------------------------------------------------------
// Launcher
// ----------------------------------------------------------------------------

void launch_custom_mmq(
        const void *          w,
        size_t                /*w_row_bytes*/,
        const void *          act_scratch,
        float *               dst,
        int                   M,
        int                   N,
        siglip2_cuda_stream_t s)
{
    cudaStream_t stream = (cudaStream_t) s;

    // Default: (128 N, 64 M), 4 warps as 4×1 → each warp does 16 mma/K-block
    // (2 N-subs × 8 M-subs). 128-thread blocks → up to 8 blocks/SM by thread
    // count, 4-5 by shmem (~6 KB each).
    constexpr int N_BLOCK = 128;
    constexpr int M_BLOCK = 64;
    constexpr int WARPS_N = 4;
    constexpr int WARPS_M = 1;
    using Cfg = mmq_config<N_BLOCK, M_BLOCK, WARPS_N, WARPS_M>;

    const int8_t * act_qs = (const int8_t *) act_scratch;
    const float  * act_d  = (const float  *) ((const char *) act_scratch + (size_t) M * K_FIXED);

    dim3 grid((N + N_BLOCK - 1) / N_BLOCK, (M + M_BLOCK - 1) / M_BLOCK, 1);
    dim3 block(32, Cfg::NWARPS, 1);

    const size_t smem_bytes =
        (size_t) Cfg::W_INTS * sizeof(int)
      + (size_t) Cfg::X_INTS * sizeof(int)
      + (size_t) N_BLOCK     * sizeof(float)
      + (size_t) M_BLOCK     * sizeof(float);

    custom_mmq_kernel<N_BLOCK, M_BLOCK, WARPS_N, WARPS_M, /*PROFILE=*/false>
        <<<grid, block, smem_bytes, stream>>>(
            (const block_q8_0 *) w, act_qs, act_d, dst, M, N);
}

void launch_custom_mmq_profile(
        const void *          w,
        size_t                /*w_row_bytes*/,
        const void *          act_scratch,
        float *               dst,
        int                   M,
        int                   N,
        void *                out_cycles_buf,
        int *                 out_grid_n,
        int *                 out_grid_m,
        siglip2_cuda_stream_t s)
{
    cudaStream_t stream = (cudaStream_t) s;

    constexpr int N_BLOCK = 128;
    constexpr int M_BLOCK = 64;
    constexpr int WARPS_N = 4;
    constexpr int WARPS_M = 1;
    using Cfg = mmq_config<N_BLOCK, M_BLOCK, WARPS_N, WARPS_M>;

    const int8_t * act_qs = (const int8_t *) act_scratch;
    const float  * act_d  = (const float  *) ((const char *) act_scratch + (size_t) M * K_FIXED);

    const int gn = (N + N_BLOCK - 1) / N_BLOCK;
    const int gm = (M + M_BLOCK - 1) / M_BLOCK;
    if (out_grid_n) *out_grid_n = gn;
    if (out_grid_m) *out_grid_m = gm;

    dim3 grid(gn, gm, 1);
    dim3 block(32, Cfg::NWARPS, 1);

    const size_t smem_bytes =
        (size_t) Cfg::W_INTS * sizeof(int)
      + (size_t) Cfg::X_INTS * sizeof(int)
      + (size_t) N_BLOCK     * sizeof(float)
      + (size_t) M_BLOCK     * sizeof(float);

    custom_mmq_kernel<N_BLOCK, M_BLOCK, WARPS_N, WARPS_M, /*PROFILE=*/true>
        <<<grid, block, smem_bytes, stream>>>(
            (const block_q8_0 *) w, act_qs, act_d, dst, M, N,
            (ProfileCycles *) out_cycles_buf);
}

}  // namespace siglip2_custom_mmq
