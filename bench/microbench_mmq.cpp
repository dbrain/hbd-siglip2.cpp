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

}  // namespace

int main(int /*argc*/, char ** /*argv*/) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed\n");
        return 1;
    }

    char devname[256] = {0};
    ggml_backend_cuda_get_device_description(0, devname, sizeof(devname));
    printf("Device: %s\n", devname);
    printf("Warmup: %d iters    Measure: %d iters\n", WARMUP_ITERS, MEASURE_ITERS);
    printf("\n");

    printf("%-32s | %10s %10s | %10s %10s | %8s | %s\n",
           "shape (M K N)", "ggml p50", "ggml GFLP", "cust p50", "cust/ggml", "cosine", "maxabs");
    printf("%-32s + %10s %10s + %10s %10s + %8s + %s\n",
           "--------------------------------",
           "----------", "----------", "----------", "----------",
           "--------", "----------");

    for (const auto & s : SHAPES) {
        Result rg, rc;
        std::vector<float>   host_x_f32;
        std::vector<uint8_t> host_w_quant;
        std::vector<float>   ref_dst;
        if (!run_ggml_path(backend, s, rg, ref_dst, host_x_f32, host_w_quant)) {
            printf("%-32s  GGML FAILED\n", s.label);
            continue;
        }
        run_custom_path(s, host_x_f32, host_w_quant, ref_dst, rc);

        const double flops = 2.0 * (double)s.M * (double)s.K * (double)s.N;
        const double ggflops = flops / (rg.p50 * 1e-3) / 1e9;

        if (rc.ran) {
            const double ratio = rc.p50 / rg.p50;
            printf("%-32s | %10.3f %10.1f | %10.3f %10.3fx | %8.6f | %.3e\n",
                   s.label, rg.p50, ggflops, rc.p50, ratio, rc.cosine, rc.max_abs);
        } else {
            printf("%-32s | %10.3f %10.1f | %10s %10s | %8s | %s\n",
                   s.label, rg.p50, ggflops, "—", "—", "—", "—");
        }
    }

    ggml_backend_free(backend);
    return 0;
}
