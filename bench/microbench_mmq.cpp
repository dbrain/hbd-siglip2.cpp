// Phase C — microbench for Q8_0×F32 (and F16×F32) mul_mat at the exact
// (M, K, N) shapes siglip2 hits per encoder block.
//
// Two columns measured back-to-back per shape:
//   1. ggml's path  — whatever ggml_mul_mat dispatches at this device + shape
//      (mma m16n8k32 for Q8_0/M≥48 on Ampere+; cuBLAS HGEMM for F16).
//   2. Custom kernel — siglip2_custom_mmq's hand-rolled mma path. Only fires
//      when K=1152 and W type is Q8_0 (the only shape it supports right now);
//      other rows print "—" in the custom column.
//
// Correctness: after warmup, snapshot the F32 output of each path into host
// vectors and report cosine + max-abs-error of custom-vs-ggml. Output is
// quantization-noise (both paths quantize activations to int8 internally),
// so the bar is "near-bit-clean" — within reduction-tree shuffle noise of
// each other.
//
// Build target: bench/microbench_mmq. Run:
//   ./build-cuda/microbench_mmq

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include "siglip2_custom_mmq.cuh"
#include "microbench_cublaslt_helpers.cuh"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Shape {
    const char *  label;
    ggml_type     w_type;
    int64_t       M;
    int64_t       K;
    int64_t       N;
};

// 9 Q8_0 shapes (3 weight × 3 M) + 3 F16 ffn_down shapes.
const std::vector<Shape> SHAPES = {
    // attn_qkv  (per-block, weight Q8_0 [1152, 3456])
    {"attn_qkv  M=64  (text n=1)",  GGML_TYPE_Q8_0,  64, 1152, 3456},
    {"attn_qkv  M=320 (text n=5)",  GGML_TYPE_Q8_0, 320, 1152, 3456},
    {"attn_qkv  M=729 (vision)",    GGML_TYPE_Q8_0, 729, 1152, 3456},
    // attn_o    (per-block, weight Q8_0 [1152, 1152])
    {"attn_o    M=64  (text n=1)",  GGML_TYPE_Q8_0,  64, 1152, 1152},
    {"attn_o    M=320 (text n=5)",  GGML_TYPE_Q8_0, 320, 1152, 1152},
    {"attn_o    M=729 (vision)",    GGML_TYPE_Q8_0, 729, 1152, 1152},
    // ffn_up    (per-block, weight Q8_0 [1152, 4304])
    {"ffn_up    M=64  (text n=1)",  GGML_TYPE_Q8_0,  64, 1152, 4304},
    {"ffn_up    M=320 (text n=5)",  GGML_TYPE_Q8_0, 320, 1152, 4304},
    {"ffn_up    M=729 (vision)",    GGML_TYPE_Q8_0, 729, 1152, 4304},
    // ffn_down  (per-block, weight F16  [4304, 1152])
    {"ffn_down  M=64  (text n=1)",  GGML_TYPE_F16,   64, 4304, 1152},
    {"ffn_down  M=320 (text n=5)",  GGML_TYPE_F16,  320, 4304, 1152},
    {"ffn_down  M=729 (vision)",    GGML_TYPE_F16,  729, 4304, 1152},
};

constexpr int  WARMUP_ITERS  = 20;
constexpr int  MEASURE_ITERS = 200;

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)std::round((v.size() - 1) * p);
    return v[idx];
}

double mean(const std::vector<double> & v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s / (double)v.size();
}

struct Result {
    bool   ran      = false;
    double p50      = 0.0;
    double p99      = 0.0;
    double mn       = 0.0;
    double mean_ms  = 0.0;
    double cosine   = 0.0;   // custom only
    double max_abs  = 0.0;   // custom only
};

bool run_ggml_path(ggml_backend_t backend, const Shape & s,
                   Result & out, std::vector<float> & ref_dst,
                   std::vector<float> & host_x_f32,
                   std::vector<uint8_t> & host_w_quant) {
    const size_t mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    std::vector<uint8_t> mem(mem_size);
    ggml_init_params ip = { mem_size, mem.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w = ggml_new_tensor_2d(ctx, s.w_type, s.K, s.N);
    ggml_set_name(w, "w");
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.K, s.M);
    ggml_set_name(x, "x");

    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_set_name(y, "y");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "  alloc_ctx_tensors failed\n");
        ggml_free(ctx);
        return false;
    }

    // Fill weight with deterministic random F32, then quantize/cast.
    {
        std::mt19937 rng(0x1eaf);
        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
        const size_t n_w = (size_t)s.K * (size_t)s.N;
        std::vector<float> wf(n_w);
        for (auto & v : wf) v = dist(rng);

        if (s.w_type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(w, wf.data(), 0, n_w * sizeof(float));
        } else if (s.w_type == GGML_TYPE_F16) {
            std::vector<uint16_t> wq(n_w);
            for (size_t i = 0; i < n_w; ++i) wq[i] = ggml_fp32_to_fp16(wf[i]);
            ggml_backend_tensor_set(w, wq.data(), 0, n_w * sizeof(uint16_t));
            host_w_quant.assign((uint8_t *) wq.data(), (uint8_t *) wq.data() + wq.size() * sizeof(uint16_t));
        } else {
            const size_t row_size = ggml_row_size(s.w_type, s.K);
            std::vector<uint8_t> wq(row_size * (size_t)s.N);
            ggml_quantize_chunk(s.w_type, wf.data(), wq.data(), 0, s.N, s.K, nullptr);
            ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
            host_w_quant = std::move(wq);
        }
    }
    {
        std::mt19937 rng(0xb01d);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        const size_t n_x = (size_t)s.K * (size_t)s.M;
        host_x_f32.resize(n_x);
        for (auto & v : host_x_f32) v = dist(rng);
        ggml_backend_tensor_set(x, host_x_f32.data(), 0, n_x * sizeof(float));
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc || !ggml_gallocr_reserve(galloc, gf) || !ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "  gallocr setup failed\n");
        if (galloc) ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    for (int i = 0; i < WARMUP_ITERS; ++i) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    // Snapshot ggml's reference output (post-warmup) so we can compare custom.
    ref_dst.resize((size_t) s.N * (size_t) s.M);
    ggml_backend_tensor_get(y, ref_dst.data(), 0, ref_dst.size() * sizeof(float));

    std::vector<double> samples;
    samples.reserve(MEASURE_ITERS);
    using clk = std::chrono::steady_clock;
    for (int i = 0; i < MEASURE_ITERS; ++i) {
        auto t0 = clk::now();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto t1 = clk::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    out.ran      = true;
    out.p50      = percentile(samples, 0.5);
    out.p99      = percentile(samples, 0.99);
    out.mn       = *std::min_element(samples.begin(), samples.end());
    out.mean_ms  = mean(samples);

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Run the custom kernel against the same inputs the ggml path used. Compare
// outputs; time per-iter same as ggml.
bool run_custom_path(const Shape & s,
                     const std::vector<float> & host_x_f32,
                     const std::vector<uint8_t> & host_w_quant,
                     const std::vector<float> & ref_dst,
                     Result & out) {
    if (s.w_type != GGML_TYPE_Q8_0 || s.K != siglip2_custom_mmq::K_FIXED) {
        out.ran = false;
        return true;  // not supported by custom kernel yet
    }

    // Allocate device buffers manually (separate from ggml's backend pool to
    // keep the comparison clean — same as ggml path's input data, but on
    // pointers we own so we don't have to interleave with ggml's gallocr).
    void *  d_w        = nullptr;
    void *  d_x        = nullptr;
    void *  d_act_scr  = nullptr;
    float * d_dst      = nullptr;

    const size_t w_bytes  = host_w_quant.size();
    const size_t x_bytes  = host_x_f32.size() * sizeof(float);
    const size_t scr_bytes = siglip2_custom_mmq::activation_scratch_bytes((int) s.M);
    const size_t dst_bytes = (size_t) s.N * (size_t) s.M * sizeof(float);

    cudaMalloc(&d_w,       w_bytes);
    cudaMalloc(&d_x,       x_bytes);
    cudaMalloc(&d_act_scr, scr_bytes);
    cudaMalloc(&d_dst,     dst_bytes);
    cudaMemcpy(d_w, host_w_quant.data(), w_bytes,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, host_x_f32.data(),   x_bytes,  cudaMemcpyHostToDevice);

    cudaStream_t stream = 0;
    siglip2_cuda_stream_t s_stream = (siglip2_cuda_stream_t) stream;

    auto run_one = [&]() {
        siglip2_custom_mmq::launch_quantize_activation((const float *) d_x, d_act_scr, (int) s.M, s_stream);
        siglip2_custom_mmq::launch_custom_mmq(d_w, /*w_row_bytes*/ 0, d_act_scr, d_dst,
                                              (int) s.M, (int) s.N, s_stream);
    };

    // Warmup.
    for (int i = 0; i < WARMUP_ITERS; ++i) run_one();
    cudaStreamSynchronize(stream);

    // Correctness snapshot.
    std::vector<float> custom_dst((size_t) s.N * (size_t) s.M);
    cudaMemcpy(custom_dst.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "  custom kernel error: %s\n", cudaGetErrorString(err));
        out.ran = false;
        cudaFree(d_w); cudaFree(d_x); cudaFree(d_act_scr); cudaFree(d_dst);
        return false;
    }

    // Cosine + max-abs.
    double dot = 0.0, n_a = 0.0, n_b = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < custom_dst.size(); ++i) {
        double a = custom_dst[i];
        double b = ref_dst[i];
        dot += a * b;
        n_a += a * a;
        n_b += b * b;
        max_abs = std::max(max_abs, std::fabs(a - b));
    }
    out.cosine  = (n_a > 0 && n_b > 0) ? dot / std::sqrt(n_a * n_b) : 0.0;
    out.max_abs = max_abs;

    // Measure.
    using clk = std::chrono::steady_clock;
    std::vector<double> samples;
    samples.reserve(MEASURE_ITERS);
    for (int i = 0; i < MEASURE_ITERS; ++i) {
        auto t0 = clk::now();
        run_one();
        cudaStreamSynchronize(stream);
        auto t1 = clk::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    out.ran     = true;
    out.p50     = percentile(samples, 0.5);
    out.p99     = percentile(samples, 0.99);
    out.mn      = *std::min_element(samples.begin(), samples.end());
    out.mean_ms = mean(samples);

    cudaFree(d_w); cudaFree(d_x); cudaFree(d_act_scr); cudaFree(d_dst);
    return true;
}

// ----------------------------------------------------------------------------
// cuBLASLt int8 path
// ----------------------------------------------------------------------------
//
// Goal: settle "does cuBLASLt int8 IMMA on Ampere beat ggml-mma at our shapes"
// before doing the full weight-packing + op_hook integration work (per
// HANDOFF-perf-next.md target 3 gating step). The runtime weight format here
// is per-row int8 with one fp32 scale per row of W and one fp32 scale per row
// of X; the matmul is cuBLASLt's int32 IMMA accumulator, then a custom
// rescale kernel multiplies the (m,n) cell by W_scale[n] * X_scale[m].
//
// Per-row scaling LOSES Q8_0's per-32-element block fidelity by design — the
// cosine reported here will fail the 0.999 floor on most shapes. That's
// expected: this column is a SPEED upper bound. If even this loose form
// can't beat ggml-mma, the per-K-tile scaling form (which is more expensive
// — multiple sub-GEMMs or a scale-aware custom output kernel) won't either,
// and target 3 is dead. If it DOES beat ggml-mma, the next step is to put
// a faithful per-block-scale path in place (likely K-blocked sub-GEMMs or
// dequant-to-bf16 + cuBLASLt HGEMM, depending on which is faster).

#define CHECK_CUBLAS(call) do {                                              \
    auto _s = (call);                                                        \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                       \
        fprintf(stderr, "cuBLASLt error %d at %s:%d (%s)\n",                 \
                (int) _s, __FILE__, __LINE__, #call);                        \
        return false;                                                        \
    }                                                                        \
} while (0)

// Repack Q8_0 weight rows to row-wise int8 + per-row fp32 scale. Q8_0 stores
// 32 elements per block, 36 blocks per K=1152, each with an fp16 scale. We
// dequant to fp32 then re-quantize per-row. Matches what the runtime would
// have to do; runs once at "load time".
static void repack_q8_to_rowscale_int8(const std::vector<uint8_t> & q8,
                                       int K, int N,
                                       std::vector<int8_t> & out_int8,
                                       std::vector<float>  & out_row_scales) {
    const int blocks_per_row = K / 32;  // Q8_0 block size is 32
    const size_t block_bytes = sizeof(uint16_t) /*scale*/ + 32 /*qs*/;
    out_int8.assign((size_t) N * (size_t) K, 0);
    out_row_scales.assign((size_t) N, 0.0f);

    std::vector<float> row_f32(K);
    for (int n = 0; n < N; ++n) {
        const uint8_t * row = q8.data() + (size_t) n * blocks_per_row * block_bytes;
        // Dequant the row.
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t * blk = row + (size_t) b * block_bytes;
            uint16_t s_u16; std::memcpy(&s_u16, blk, sizeof(uint16_t));
            const float scale = ggml_fp16_to_fp32(s_u16);
            const int8_t * qs = (const int8_t *)(blk + sizeof(uint16_t));
            for (int i = 0; i < 32; ++i) {
                row_f32[b * 32 + i] = scale * (float) qs[i];
            }
        }
        // Per-row scale = max(|row|) / 127.
        float maxabs = 0.0f;
        for (int k = 0; k < K; ++k) maxabs = std::max(maxabs, std::fabs(row_f32[k]));
        const float scale = maxabs > 0.0f ? maxabs / 127.0f : 1.0f;
        const float inv   = 1.0f / scale;
        for (int k = 0; k < K; ++k) {
            int v = (int) std::lround(row_f32[k] * inv);
            v = std::max(-127, std::min(127, v));
            out_int8[(size_t) n * (size_t) K + (size_t) k] = (int8_t) v;
        }
        out_row_scales[n] = scale;
    }
}

// Run the cuBLASLt int8 path for one shape. Reuses host_x_f32, host_w_quant
// from the ggml run; computes its own packings.
static bool run_cublaslt_path(cublasLtHandle_t lt,
                              const Shape & s,
                              const std::vector<float>   & host_x_f32,
                              const std::vector<uint8_t> & host_w_quant,
                              const std::vector<float>   & ref_dst,
                              Result & out) {
    if (s.w_type != GGML_TYPE_Q8_0) {
        out.ran = false;
        return true;  // not applicable
    }

    // 1) Pack weights: Q8_0 → row-wise int8 [N, K] + fp32 scale [N].
    std::vector<int8_t> w_int8;
    std::vector<float>  w_scales;
    repack_q8_to_rowscale_int8(host_w_quant, (int) s.K, (int) s.N, w_int8, w_scales);

    // 2) Allocate device buffers.
    int8_t  * d_w        = nullptr;
    int8_t  * d_x_int8   = nullptr;
    int32_t * d_dst_int32 = nullptr;
    float   * d_dst_f32  = nullptr;
    float   * d_w_scales = nullptr;
    float   * d_x_scales = nullptr;
    float   * d_x_f32    = nullptr;
    cudaMalloc(&d_w,          w_int8.size());
    cudaMalloc(&d_x_int8,     (size_t) s.M * s.K);
    cudaMalloc(&d_dst_int32,  (size_t) s.M * s.N * sizeof(int32_t));
    cudaMalloc(&d_dst_f32,    (size_t) s.M * s.N * sizeof(float));
    cudaMalloc(&d_w_scales,   (size_t) s.N * sizeof(float));
    cudaMalloc(&d_x_scales,   (size_t) s.M * sizeof(float));
    cudaMalloc(&d_x_f32,      host_x_f32.size() * sizeof(float));
    cudaMemcpy(d_w,        w_int8.data(),    w_int8.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_scales, w_scales.data(),  w_scales.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_f32,    host_x_f32.data(), host_x_f32.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream = 0;

    // 3) Quantize x: F32 [K, M] → int8 [M, K] row-major + per-row scale [M].
    launch_quantize_x_to_rowscale_int8(d_x_f32, d_x_int8, d_x_scales,
                                       (int) s.M, (int) s.K, stream);

    // 4) cuBLASLt int8 IMMA setup.
    //
    // We compute Y[M, N] = X[M, K] · W[N, K]^T. cuBLASLt's strict layout is
    // column-major; we lay out X and W in row-major K-fastest, treat them
    // as transposed cm matrices (K rows, M/N cols) and use the OP_T flag.
    //
    // Op shape (column-major):
    //   A: int8 K x N  (which is W's row-major [N, K] viewed transposed)
    //   B: int8 K x M  (which is X's row-major [M, K] viewed transposed)
    //   C: int32 N x M (column-major)
    // We swap A/B so the column-major output is N x M, then read it as M x N
    // row-major below for the rescale.

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr, c_layout = nullptr;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    cublasOperation_t op_t = CUBLAS_OP_T;
    cublasOperation_t op_n = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

    // A = W_int8, row-major [N, K] → cm view [K, N], ld=K
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_8I, s.K, s.N, s.K));
    // B = X_int8, row-major [M, K] → cm view [K, M], ld=K
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_8I, s.K, s.M, s.K));
    // C = int32 cm [N, M], ld=N — we then transpose-read as row-major [M, N]
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_32I, s.N, s.M, s.N));

    int32_t alpha = 1, beta = 0;

    auto do_gemm = [&]() {
        return cublasLtMatmul(lt, op_desc,
                              &alpha,
                              d_w,        a_layout,
                              d_x_int8,   b_layout,
                              &beta,
                              d_dst_int32, c_layout,
                              d_dst_int32, c_layout,
                              /*algo=*/   nullptr,
                              /*workspace=*/ nullptr, /*ws_bytes=*/ 0,
                              stream);
    };

    auto run_one = [&]() {
        // Re-quantize x each iteration so the timing reflects what the runtime
        // will pay (the activation is fresh per encode).
        launch_quantize_x_to_rowscale_int8(d_x_f32, d_x_int8, d_x_scales,
                                           (int) s.M, (int) s.K, stream);
        do_gemm();
        // int32 → fp32 in-place rescale. cuBLASLt wrote int32 [N, M] cm; we
        // multiply by w_scale[n] * x_scale[m] cell-by-cell and write fp32 at
        // the same address (sizeof matches). The correctness probe re-runs
        // and reads int32 to host without this kernel so we can compute
        // cosine at full precision.
        launch_int32_rescale_to_fp32_inplace(d_dst_int32, d_w_scales, d_x_scales,
                                             (int) s.M, (int) s.N, stream);
    };

    // Warmup.
    for (int i = 0; i < WARMUP_ITERS; ++i) run_one();
    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "  cublaslt warmup error: %s\n", cudaGetErrorString(err));
        out.ran = false;
        cublasLtMatrixLayoutDestroy(a_layout);
        cublasLtMatrixLayoutDestroy(b_layout);
        cublasLtMatrixLayoutDestroy(c_layout);
        cublasLtMatmulDescDestroy(op_desc);
        cudaFree(d_w); cudaFree(d_x_int8); cudaFree(d_dst_int32);
        cudaFree(d_dst_f32); cudaFree(d_w_scales); cudaFree(d_x_scales); cudaFree(d_x_f32);
        return false;
    }

    // Correctness probe — convert int32→fp32, then apply the rescale and
    // compare against ref_dst (which is column-major [N, M] from ggml's
    // mul_mat). A separate clean run since the in-place rescale above
    // overwrote d_dst_int32 with floats.
    {
        // Re-do the gemm and convert to fp32 outside the timing loop.
        launch_quantize_x_to_rowscale_int8(d_x_f32, d_x_int8, d_x_scales,
                                           (int) s.M, (int) s.K, stream);
        do_gemm();
        // int32 cm[N, M] → fp32 cm[N, M] (same shape as ref_dst, no transpose).
        std::vector<int32_t> raw_int32((size_t) s.M * (size_t) s.N);
        cudaMemcpy(raw_int32.data(), d_dst_int32, raw_int32.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        std::vector<float> x_scales_h((size_t) s.M);
        cudaMemcpy(x_scales_h.data(), d_x_scales, x_scales_h.size() * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> our_dst((size_t) s.M * (size_t) s.N);
        // ref_dst is what ggml_mul_mat(W, X) produced — shape from
        // ggml's perspective is (N, M) column-major (W is K x N, X is K x M,
        // result is N x M). Our raw_int32 is also cm [N, M]. So index match
        // is: cell(n, m) at i = m * N + n.
        double dot = 0.0, na = 0.0, nb = 0.0, max_abs = 0.0;
        for (int m = 0; m < (int) s.M; ++m) {
            const float xs = x_scales_h[m];
            for (int n = 0; n < (int) s.N; ++n) {
                const size_t i = (size_t) m * s.N + n;
                const float v = (float) raw_int32[i] * xs * w_scales[n];
                our_dst[i] = v;
                const float r = ref_dst[i];
                dot += v * r; na += v * v; nb += r * r;
                max_abs = std::max(max_abs, (double) std::fabs(v - r));
            }
        }
        out.cosine  = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
        out.max_abs = max_abs;
    }

    // Measure.
    using clk = std::chrono::steady_clock;
    std::vector<double> samples;
    samples.reserve(MEASURE_ITERS);
    for (int i = 0; i < MEASURE_ITERS; ++i) {
        auto t0 = clk::now();
        run_one();
        cudaStreamSynchronize(stream);
        auto t1 = clk::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    out.ran     = true;
    out.p50     = percentile(samples, 0.5);
    out.p99     = percentile(samples, 0.99);
    out.mn      = *std::min_element(samples.begin(), samples.end());
    out.mean_ms = mean(samples);

    cublasLtMatrixLayoutDestroy(a_layout);
    cublasLtMatrixLayoutDestroy(b_layout);
    cublasLtMatrixLayoutDestroy(c_layout);
    cublasLtMatmulDescDestroy(op_desc);
    cudaFree(d_w); cudaFree(d_x_int8); cudaFree(d_dst_int32);
    cudaFree(d_dst_f32); cudaFree(d_w_scales); cudaFree(d_x_scales); cudaFree(d_x_f32);
    return true;
}

}  // namespace

// One-shot profile of the custom kernel at the cfe shape (M=320). Dumps
// per-region cycle breakdown averaged across CUDA blocks. Returns nothing
// useful — this is purely for understanding where the time goes.
static void run_profile_cfe(const Shape & s,
                            const std::vector<float> & host_x_f32,
                            const std::vector<uint8_t> & host_w_quant)
{
    using namespace siglip2_custom_mmq;

    if (s.w_type != GGML_TYPE_Q8_0 || s.K != K_FIXED) {
        printf("[profile] shape unsupported, skipping\n");
        return;
    }

    void * d_w = nullptr; void * d_x = nullptr; void * d_act = nullptr;
    float * d_dst = nullptr; void * d_cyc = nullptr;
    cudaMalloc(&d_w,   host_w_quant.size());
    cudaMalloc(&d_x,   host_x_f32.size() * sizeof(float));
    cudaMalloc(&d_act, activation_scratch_bytes((int) s.M));
    cudaMalloc(&d_dst, (size_t) s.N * (size_t) s.M * sizeof(float));
    // Worst-case 2048 blocks (cfe = 27 × 5 = 135).
    const int max_blocks = 2048;
    cudaMalloc(&d_cyc, (size_t) max_blocks * sizeof(ProfileCycles));
    cudaMemset(d_cyc, 0, (size_t) max_blocks * sizeof(ProfileCycles));
    cudaMemcpy(d_w, host_w_quant.data(), host_w_quant.size(),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, host_x_f32.data(),   host_x_f32.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream = 0;
    auto s_stream = (siglip2_cuda_stream_t) stream;

    // Warmup (without profile counters, so the first JIT/L1 hit doesn't
    // contaminate the profiled run).
    for (int i = 0; i < 5; ++i) {
        launch_quantize_activation((const float *) d_x, d_act, (int) s.M, s_stream);
        launch_custom_mmq(d_w, 0, d_act, d_dst, (int) s.M, (int) s.N, s_stream);
    }
    cudaStreamSynchronize(stream);

    // Profile run.
    int grid_n = 0, grid_m = 0;
    launch_quantize_activation((const float *) d_x, d_act, (int) s.M, s_stream);
    launch_custom_mmq_profile(d_w, 0, d_act, d_dst, (int) s.M, (int) s.N,
                              d_cyc, &grid_n, &grid_m, s_stream);
    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[profile] kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_w); cudaFree(d_x); cudaFree(d_act); cudaFree(d_dst); cudaFree(d_cyc);
        return;
    }

    const int n_blocks = grid_n * grid_m;
    std::vector<ProfileCycles> host_cyc(n_blocks);
    cudaMemcpy(host_cyc.data(), d_cyc,
               (size_t) n_blocks * sizeof(ProfileCycles),
               cudaMemcpyDeviceToHost);

    // Aggregate.
    unsigned long long sum_load = 0, sum_sync1 = 0, sum_compute = 0, sum_sync2 = 0;
    unsigned long long min_load = ~0ull, max_load = 0;
    unsigned long long min_compute = ~0ull, max_compute = 0;
    for (const auto & c : host_cyc) {
        sum_load    += c.load;
        sum_sync1   += c.sync1;
        sum_compute += c.compute;
        sum_sync2   += c.sync2;
        if (c.load    < min_load)    min_load    = c.load;
        if (c.load    > max_load)    max_load    = c.load;
        if (c.compute < min_compute) min_compute = c.compute;
        if (c.compute > max_compute) max_compute = c.compute;
    }
    const double inv_blocks = 1.0 / (double) n_blocks;
    const double avg_load    = sum_load    * inv_blocks;
    const double avg_sync1   = sum_sync1   * inv_blocks;
    const double avg_compute = sum_compute * inv_blocks;
    const double avg_sync2   = sum_sync2   * inv_blocks;
    const double avg_total   = avg_load + avg_sync1 + avg_compute + avg_sync2;

    printf("\n--- profile @ %s (grid=%dx%d=%d blocks, K_BLOCKS=%d) ---\n",
           s.label, grid_n, grid_m, n_blocks, (int) K_BLOCKS);
    printf("  region        avg cycles/block   share    avg cycles/K-block\n");
    printf("  load          %15.0f   %5.1f%%  %18.0f\n",
           avg_load,    100.0 * avg_load    / avg_total, avg_load    / (double) K_BLOCKS);
    printf("  sync (post-L) %15.0f   %5.1f%%  %18.0f\n",
           avg_sync1,   100.0 * avg_sync1   / avg_total, avg_sync1   / (double) K_BLOCKS);
    printf("  compute       %15.0f   %5.1f%%  %18.0f\n",
           avg_compute, 100.0 * avg_compute / avg_total, avg_compute / (double) K_BLOCKS);
    printf("  sync (end)    %15.0f   %5.1f%%  %18.0f\n",
           avg_sync2,   100.0 * avg_sync2   / avg_total, avg_sync2   / (double) K_BLOCKS);
    printf("  total         %15.0f   100.0%%  %18.0f\n",
           avg_total, avg_total / (double) K_BLOCKS);
    printf("  load   range  min=%llu max=%llu  (block-to-block variance)\n",
           min_load, max_load);
    printf("  compute range min=%llu max=%llu\n",
           min_compute, max_compute);

    cudaFree(d_w); cudaFree(d_x); cudaFree(d_act); cudaFree(d_dst); cudaFree(d_cyc);
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    bool profile_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--profile") profile_mode = true;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed\n");
        return 1;
    }

    cublasLtHandle_t lt = nullptr;
    if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasLtCreate failed\n");
        ggml_backend_free(backend);
        return 1;
    }

    char devname[256] = {0};
    ggml_backend_cuda_get_device_description(0, devname, sizeof(devname));
    printf("Device: %s\n", devname);
    printf("Warmup: %d iters    Measure: %d iters\n", WARMUP_ITERS, MEASURE_ITERS);
    printf("\n");

    printf("%-32s | %10s %10s | %10s %10s | %10s %10s %10s | %8s\n",
           "shape (M K N)",
           "ggml p50", "ggml GFLP",
           "cust p50", "cust/ggml",
           "cuBLLt p50", "cuBLLt/gg", "cuBLLt cos",
           "cust cos");
    printf("%-32s + %10s %10s + %10s %10s + %10s %10s %10s + %8s\n",
           "--------------------------------",
           "----------", "----------", "----------", "----------",
           "----------", "----------", "----------", "--------");

    for (const auto & s : SHAPES) {
        Result rg, rc, rl;
        std::vector<float>   host_x_f32;
        std::vector<uint8_t> host_w_quant;
        std::vector<float>   ref_dst;
        if (!run_ggml_path(backend, s, rg, ref_dst, host_x_f32, host_w_quant)) {
            printf("%-32s  GGML FAILED\n", s.label);
            continue;
        }
        run_custom_path(s, host_x_f32, host_w_quant, ref_dst, rc);
        run_cublaslt_path(lt, s, host_x_f32, host_w_quant, ref_dst, rl);

        const double flops = 2.0 * (double)s.M * (double)s.K * (double)s.N;
        const double ggflops = flops / (rg.p50 * 1e-3) / 1e9;

        // ggml + cuBLASLt + custom row.
        printf("%-32s | %10.3f %10.1f", s.label, rg.p50, ggflops);
        if (rc.ran) {
            printf(" | %10.3f %9.3fx", rc.p50, rc.p50 / rg.p50);
        } else {
            printf(" | %10s %10s", "—", "—");
        }
        if (rl.ran) {
            printf(" | %10.3f %9.3fx %10.6f",
                   rl.p50, rl.p50 / rg.p50, rl.cosine);
        } else {
            printf(" | %10s %10s %10s", "—", "—", "—");
        }
        if (rc.ran) {
            printf(" | %8.6f\n", rc.cosine);
        } else {
            printf(" | %8s\n", "—");
        }

        // Profile mode: run the cycle breakdown for the cfe shape after
        // benching it.
        if (profile_mode && std::string(s.label).find("M=320 (text n=5)") != std::string::npos) {
            run_profile_cfe(s, host_x_f32, host_w_quant);
        }
    }

    cublasLtDestroy(lt);
    ggml_backend_free(backend);
    return 0;
}
