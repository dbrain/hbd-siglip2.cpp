// siglip2 custom MMQ — Phase C kernel-lap entry point.
//
// v6: cp.async + double-buffered shmem. cp.async.cg.shared.global is sm_80+
// (Ampere) and lets each thread queue an async global→shmem copy that the
// SM can run concurrently with mma compute — proper memory-latency hiding
// over the K-loop, the canonical "you've hit the compute path, now feed it"
// optimization.
//
// Pipeline shape:
//   prologue: issue loads for K[0] into buf 0; commit.
//   for kb in 1..K_BLOCKS-1:
//       issue loads for K[kb] into buf (kb%2); commit.
//       wait_prior(1)  // K[kb-1] loads done
//       __syncthreads()
//       compute K[kb-1] from buf ((kb-1)%2)
//   epilogue:
//       wait_prior(0); __syncthreads()
//       compute K[K_BLOCKS-1] from buf ((K_BLOCKS-1)%2)
//
// Convention (matches ggml_mul_mat semantics):
//   W:   (K, N)   Q8_0   K innermost; stored as N rows × K_BLOCKS of block_q8_0
//   x:   (K, M)   F32    K innermost; quantized on the fly to (act_qs, act_d)
//   dst: (N, M)   F32    N innermost; dst[m * N + n]  is the (n, m) entry

#include "siglip2_custom_mmq.cuh"

#include "ggml-cuda/mma.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cuda_fp16.h>
#include <cuda_pipeline.h>
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
// cp.async helpers — manual asm to keep the load shape consistent with our
// per-thread coalesced int-stride pattern. CUDA's __pipeline_memcpy_async
// also works but adds a pipeline_t object we'd otherwise carry around.
// ----------------------------------------------------------------------------

__device__ __forceinline__ void cp_async_4(int * dst_smem, const void * src_gmem) {
#if __CUDA_ARCH__ >= 800
    unsigned smem_addr = __cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :: "r"(smem_addr), "l"(src_gmem));
#else
    *dst_smem = *(const int *) src_gmem;
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// ----------------------------------------------------------------------------
// Custom MMQ kernel — v6.
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
    // Per-buf: W_INTS + X_INTS + N_BLOCK dw + M_BLOCK dx.
    static constexpr int BUF_INTS = W_INTS + X_INTS + N_BLOCK + M_BLOCK; // floats fit in ints (same width)
};

// Async-issue all loads for K-block kb into the shmem buffers given by their
// base int-pointers. The buffers are sized for exactly one K-block (W_INTS
// ints + X_INTS ints + N_BLOCK floats dw + M_BLOCK floats dx).
template <int N_BLOCK, int M_BLOCK, int WARPS_N_, int WARPS_M_>
__device__ __forceinline__ void issue_kb_loads(
        const block_q8_0 * w,
        const int8_t     * act_qs,
        const float      * act_d,
        int *   smem_w,
        int *   smem_x,
        float * smem_dw,
        float * smem_dx,
        int kb,
        int block_n0, int block_m0,
        int M, int N,
        int tid, int total_threads)
{
    using Cfg = mmq_config<N_BLOCK, M_BLOCK, WARPS_N_, WARPS_M_>;

    // W qs.
#pragma unroll
    for (int i = tid; i < Cfg::W_INTS; i += total_threads) {
        const int n_row    = i / 8;
        const int j        = i % 8;
        const int n_global = block_n0 + n_row;
        // Q8_0 qs sits at +2 bytes (after half d). cp.async needs 4-byte
        // aligned source — and 2*j ints into qs is 2 + j*4 bytes from the
        // block start, which is 2-byte aligned. Fall back to get_int_b2 for
        // this read (sync, scalar) and stash into shmem normally.
        int v = 0;
        if (n_global < N) {
            const block_q8_0 * blk = w + n_global * K_BLOCKS + kb;
            v = get_int_b2(blk->qs, j);
        }
        smem_w[i] = v;
    }

    // W scales (dw): one half per row, read & convert sync.
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

    // x qs — fully aligned 4-byte loads. cp.async eligible.
#pragma unroll
    for (int i = tid; i < Cfg::X_INTS; i += total_threads) {
        const int m_row    = i / 8;
        const int j        = i % 8;
        const int m_global = block_m0 + m_row;
        if (m_global < M) {
            const int8_t * row = act_qs + m_global * K_FIXED + kb * K_BLOCK_SIZE;
            cp_async_4(&smem_x[i], &((const int *) row)[j]);
        } else {
            smem_x[i] = 0;
        }
    }

    // x scales (dx): float per row, also aligned.
#pragma unroll
    for (int i = tid; i < M_BLOCK; i += total_threads) {
        const int m_global = block_m0 + i;
        if (m_global < M) {
            cp_async_4((int *) &smem_dx[i], &act_d[m_global * K_BLOCKS + kb]);
        } else {
            smem_dx[i] = 0.f;
        }
    }
}

template <int N_BLOCK, int M_BLOCK, int WARPS_N_, int WARPS_M_>
__launch_bounds__(mmq_config<N_BLOCK, M_BLOCK, WARPS_N_, WARPS_M_>::THREADS, 1)
__global__ void custom_mmq_kernel(
        const block_q8_0 * __restrict__ w,
        const int8_t     * __restrict__ act_qs,
        const float      * __restrict__ act_d,
        float            * __restrict__ dst,
        int M, int N)
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

    // Per-thread scale endpoints (matches mma layout).
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
    // Two K-block buffers, ping-pong indexed by (kb % 2).
    int * buf_base[2];
    buf_base[0] = sm_data;
    buf_base[1] = sm_data + Cfg::BUF_INTS;
    auto buf_smem_w  = [&](int b) { return buf_base[b]; };
    auto buf_smem_x  = [&](int b) { return buf_base[b] + Cfg::W_INTS; };
    auto buf_smem_dw = [&](int b) { return (float *) (buf_base[b] + Cfg::W_INTS + Cfg::X_INTS); };
    auto buf_smem_dx = [&](int b) { return (float *) (buf_base[b] + Cfg::W_INTS + Cfg::X_INTS + N_BLOCK); };

    const int tid           = warp_id * 32 + lane;
    const int total_threads = Cfg::THREADS;

    // Prologue: issue loads for K[0] into buf 0.
    issue_kb_loads<N_BLOCK, M_BLOCK, WARPS_N_, WARPS_M_>(
        w, act_qs, act_d,
        buf_smem_w(0), buf_smem_x(0), buf_smem_dw(0), buf_smem_dx(0),
        /*kb=*/0, block_n0, block_m0, M, N, tid, total_threads);
    cp_async_commit();

    // Pipeline: iter computes K[kb], the loads for K[kb+1] are async-issued
    // first so they overlap with the K[kb] mma below.
#pragma unroll 1
    for (int kb = 0; kb < K_BLOCKS - 1; ++kb) {
        // Issue loads for next K-block while current is still in flight.
        issue_kb_loads<N_BLOCK, M_BLOCK, WARPS_N_, WARPS_M_>(
            w, act_qs, act_d,
            buf_smem_w((kb+1) & 1), buf_smem_x((kb+1) & 1),
            buf_smem_dw((kb+1) & 1), buf_smem_dx((kb+1) & 1),
            kb + 1, block_n0, block_m0, M, N, tid, total_threads);
        cp_async_commit();

        // Wait for the OLDER group (current K[kb]) to be done.
        cp_async_wait_group<1>();
        __syncthreads();

        // -- Compute K[kb] from buf (kb & 1).
        const int b = kb & 1;
        const int   * smem_w  = buf_smem_w (b);
        const int   * smem_x  = buf_smem_x (b);
        const float * smem_dw = buf_smem_dw(b);
        const float * smem_dx = buf_smem_dx(b);

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

        tile<16, 8, int> A[N_SUBTILES];
#pragma unroll
        for (int sn = 0; sn < N_SUBTILES; ++sn) {
            const int n_strip = warp_n * N_SUBTILES + sn;
            const int * A_smem = smem_w + n_strip * 16 * (K_BLOCK_SIZE / 4);
            load_ldmatrix(A[sn], A_smem, K_BLOCK_SIZE / 4);
        }

#pragma unroll
        for (int sm = 0; sm < M_SUBTILES; ++sm) {
            const int m_strip = warp_m * M_SUBTILES + sm;
            const int * B_smem = smem_x + m_strip * 8 * (K_BLOCK_SIZE / 4);
            tile<8, 8, int> B;
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
    }

    // Epilogue: K[K_BLOCKS-1] still pending; wait for everything, compute.
    cp_async_wait_group<0>();
    __syncthreads();

    {
        const int kb = K_BLOCKS - 1;
        const int b = kb & 1;
        const int   * smem_w  = buf_smem_w (b);
        const int   * smem_x  = buf_smem_x (b);
        const float * smem_dw = buf_smem_dw(b);
        const float * smem_dx = buf_smem_dx(b);

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

        tile<16, 8, int> A[N_SUBTILES];
#pragma unroll
        for (int sn = 0; sn < N_SUBTILES; ++sn) {
            const int n_strip = warp_n * N_SUBTILES + sn;
            const int * A_smem = smem_w + n_strip * 16 * (K_BLOCK_SIZE / 4);
            load_ldmatrix(A[sn], A_smem, K_BLOCK_SIZE / 4);
        }

#pragma unroll
        for (int sm = 0; sm < M_SUBTILES; ++sm) {
            const int m_strip = warp_m * M_SUBTILES + sm;
            const int * B_smem = smem_x + m_strip * 8 * (K_BLOCK_SIZE / 4);
            tile<8, 8, int> B;
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

    // Default: (128 N, 64 M), 4 warps as 4×1.
    constexpr int N_BLOCK = 128;
    constexpr int M_BLOCK = 64;
    constexpr int WARPS_N = 4;
    constexpr int WARPS_M = 1;
    using Cfg = mmq_config<N_BLOCK, M_BLOCK, WARPS_N, WARPS_M>;

    const int8_t * act_qs = (const int8_t *) act_scratch;
    const float  * act_d  = (const float  *) ((const char *) act_scratch + (size_t) M * K_FIXED);

    dim3 grid((N + N_BLOCK - 1) / N_BLOCK, (M + M_BLOCK - 1) / M_BLOCK, 1);
    dim3 block(32, Cfg::NWARPS, 1);

    // 2 K-buffers worth of shmem.
    const size_t smem_bytes = (size_t) Cfg::BUF_INTS * 2 * sizeof(int);

    custom_mmq_kernel<N_BLOCK, M_BLOCK, WARPS_N, WARPS_M>
        <<<grid, block, smem_bytes, stream>>>(
            (const block_q8_0 *) w, act_qs, act_d, dst, M, N);
}

}  // namespace siglip2_custom_mmq
