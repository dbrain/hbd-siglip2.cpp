// Microbench helpers for the cuBLASLt int8 column. Kernels live in the .cu
// sibling so the bench's main TU stays C++.
//
// quantize_x_to_rowscale_int8: F32 [K, M] (column-major K-fastest) → int8
// [M, K] (row-major K-fastest) plus one fp32 max-abs/127 scale per row of M.
//
// int32_rescale_to_fp32: read int32 [N, M] (column-major from cuBLASLt int8
// IMMA — A=W^T int8, B=X^T int8 → C=int32), multiply each cell by
// w_scale[n] * x_scale[m], write fp32 in-place at the same address (sizeof
// matches). Used for the timed run only; the correctness probe re-runs and
// reads int32 to host before applying scales for max-fidelity cosine.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

void launch_quantize_x_to_rowscale_int8(
    const float * d_x_f32,    // [K, M] column-major
    int8_t      * d_x_int8,   // [M, K] row-major
    float       * d_x_scales, // [M]
    int M, int K,
    cudaStream_t stream);

void launch_int32_rescale_to_fp32_inplace(
    int32_t     * d_dst_int32_to_fp32,  // size [N, M], reinterpreted as float on output
    const float * d_w_scales,           // [N]
    const float * d_x_scales,           // [M]
    int M, int N,
    cudaStream_t stream);
