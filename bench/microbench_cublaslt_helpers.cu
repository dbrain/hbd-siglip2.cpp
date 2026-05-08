// Kernels for the cuBLASLt int8 column of microbench_mmq. See the .cuh for
// the host API contracts.

#include "microbench_cublaslt_helpers.cuh"

#include <cuda_runtime.h>

namespace {

// One block per output row m. Two-pass: first pass finds row max-abs (warp
// reduce + shared-mem reduce); second pass writes int8 quantized values.
__global__ void quantize_x_to_rowscale_int8_kernel(
    const float * __restrict__ x,    // [K, M] column-major (k * M + m)
    int8_t      * __restrict__ q,    // [M, K] row-major (m * K + k)
    float       * __restrict__ scale,// [M]
    int M, int K)
{
    const int m = blockIdx.x;
    if (m >= M) return;
    extern __shared__ float smax[];
    const int tid = threadIdx.x;

    float local = 0.0f;
    for (int k = tid; k < K; k += blockDim.x) {
        const float v = x[(size_t) k * M + m];
        const float a = fabsf(v);
        if (a > local) local = a;
    }
    smax[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = smax[tid + s];
            if (a > smax[tid]) smax[tid] = a;
        }
        __syncthreads();
    }
    const float maxabs = smax[0];
    const float row_scale = maxabs > 0.0f ? maxabs / 127.0f : 1.0f;
    const float inv = 1.0f / row_scale;
    if (tid == 0) scale[m] = row_scale;

    for (int k = tid; k < K; k += blockDim.x) {
        const float v = x[(size_t) k * M + m];
        int qv = __float2int_rn(v * inv);
        if (qv < -127) qv = -127;
        if (qv >  127) qv =  127;
        q[(size_t) m * K + k] = (int8_t) qv;
    }
}

// Reinterpret int32 [N, M] (column-major) as float [N, M]; multiply by
// w_scale[n] * x_scale[m] in-place. Memory layout per cell (n, m) is at
// offset (m * N + n) — same address whether read as int32 or float (4
// bytes either way), so this is in-place.
__global__ void int32_rescale_to_fp32_inplace_kernel(
    int32_t     * __restrict__ buf,
    const float * __restrict__ w_scales,
    const float * __restrict__ x_scales,
    int M, int N)
{
    const int m = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;
    if (n >= N || m >= M) return;
    const size_t i = (size_t) m * N + n;
    const float s = x_scales[m] * w_scales[n];
    const int32_t v = buf[i];
    // Type-pun via reinterpret: write the float into the same 4-byte slot.
    float f = (float) v * s;
    ((float *) buf)[i] = f;
}

}  // namespace

void launch_quantize_x_to_rowscale_int8(
    const float * d_x_f32,
    int8_t      * d_x_int8,
    float       * d_x_scales,
    int M, int K,
    cudaStream_t stream)
{
    constexpr int threads = 256;
    const size_t smem    = threads * sizeof(float);
    quantize_x_to_rowscale_int8_kernel<<<M, threads, smem, stream>>>(
        d_x_f32, d_x_int8, d_x_scales, M, K);
}

void launch_int32_rescale_to_fp32_inplace(
    int32_t     * buf,
    const float * d_w_scales,
    const float * d_x_scales,
    int M, int N,
    cudaStream_t stream)
{
    constexpr int threads = 256;
    const dim3 grid((unsigned) M, (unsigned) ((N + threads - 1) / threads));
    int32_rescale_to_fp32_inplace_kernel<<<grid, threads, 0, stream>>>(
        buf, d_w_scales, d_x_scales, M, N);
}
