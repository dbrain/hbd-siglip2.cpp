#include "siglip2_preproc.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace siglip2 {

namespace {

// Snap a scaled dimension up to a multiple of patch_size, with a floor of 1 patch.
int scaled_size(double scale, int size, int patch_size) {
    double v = (double)size * scale;
    int rounded = (int)std::ceil(v / (double)patch_size) * patch_size;
    return std::max(patch_size, rounded);
}

// Binary-search the largest scale that yields num_patches <= max_num_patches.
// Mirrors HF get_image_size_for_max_num_patches.
void compute_target_size(
    int    height,
    int    width,
    int    patch_size,
    int    max_num_patches,
    int &  out_h,
    int &  out_w) {
    const double eps = 1e-5;
    double lo = eps / 10.0;
    double hi = 100.0;
    while ((hi - lo) >= eps) {
        double mid = 0.5 * (lo + hi);
        int th = scaled_size(mid, height, patch_size);
        int tw = scaled_size(mid, width,  patch_size);
        long long n = ((long long)th / patch_size) * ((long long)tw / patch_size);
        if (n <= (long long)max_num_patches) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    out_h = scaled_size(lo, height, patch_size);
    out_w = scaled_size(lo, width,  patch_size);
}

// Antialiased bilinear ("triangle") resize. Matches torchvision's
// F.interpolate(mode='bilinear', antialias=True) which HF
// Siglip2ImageProcessor uses by default for both up- and down-sampling.
//
// Algorithm (separable triangular low-pass):
//   per axis:
//     scale = src/dst
//     support_radius r = max(scale, 1.0)   // widens the kernel when downsampling
//     for each output pixel i:
//       center = (i + 0.5) * scale - 0.5
//       weight at source position k:  max(0, 1 - |(k - center) / r|)
//     normalize weights so they sum to 1
//   apply along height first into a float intermediate, then along width.
//
// Reference: PyTorch aten/src/ATen/native/cpu/UpSampleKernel.cpp
// upsample_bilinear2d_aa (same triangular separable filter).
struct AAFilter1D {
    int               max_taps = 0;
    std::vector<int>  first;        // first source index per output pixel
    std::vector<int>  taps;         // tap count per output pixel
    std::vector<float> weights;     // [out * max_taps + tap]
};

void build_aa_filter(int src_size, int dst_size, AAFilter1D & f) {
    const double scale = (double)src_size / (double)dst_size;
    const double r     = std::max(scale, 1.0);
    f.max_taps = (int)std::ceil(2.0 * r) + 2;
    f.first.assign(dst_size, 0);
    f.taps.assign(dst_size, 0);
    f.weights.assign((size_t)dst_size * f.max_taps, 0.0f);

    for (int i = 0; i < dst_size; ++i) {
        const double center = (i + 0.5) * scale - 0.5;
        int k0 = (int)std::ceil (center - r);
        int k1 = (int)std::floor(center + r);
        // r is at least 1.0, so the window has at least one tap.
        f.first[i] = k0;
        const int n = k1 - k0 + 1;

        double sum = 0.0;
        for (int t = 0; t < n; ++t) {
            const int k = k0 + t;
            const double d = (k - center) / r;
            const double w = std::max(0.0, 1.0 - std::abs(d));
            f.weights[(size_t)i * f.max_taps + t] = (float)w;
            sum += w;
        }
        // Normalize. (Sum is always > 0 because the center tap has w=1.)
        const float inv = (float)(1.0 / sum);
        for (int t = 0; t < n; ++t) {
            f.weights[(size_t)i * f.max_taps + t] *= inv;
        }
        f.taps[i] = n;
    }
}

void resize_bilinear_u8(
    const uint8_t * src,
    int             src_h,
    int             src_w,
    int             channels,
    uint8_t *       dst,
    int             dst_h,
    int             dst_w) {
    AAFilter1D fy, fx;
    build_aa_filter(src_h, dst_h, fy);
    build_aa_filter(src_w, dst_w, fx);

    // Pass 1: filter along height. Output shape: (dst_h, src_w, channels) float.
    std::vector<float> tmp((size_t)dst_h * src_w * channels);
    for (int y = 0; y < dst_h; ++y) {
        const int   k0 = fy.first[y];
        const int   n  = fy.taps[y];
        const float * wrow = fy.weights.data() + (size_t)y * fy.max_taps;
        for (int x = 0; x < src_w; ++x) {
            float * out = tmp.data() + ((size_t)y * src_w + x) * channels;
            for (int c = 0; c < channels; ++c) out[c] = 0.0f;
            for (int t = 0; t < n; ++t) {
                int sy = k0 + t;
                if (sy < 0)        sy = 0;
                if (sy >= src_h)   sy = src_h - 1;
                const float w = wrow[t];
                const uint8_t * sp = src + ((size_t)sy * src_w + x) * channels;
                for (int c = 0; c < channels; ++c) {
                    out[c] += w * (float)sp[c];
                }
            }
        }
    }

    // Pass 2: filter along width. Input: tmp (dst_h, src_w, c) float.
    // Output: dst (dst_h, dst_w, c) uint8.
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            const int   k0 = fx.first[x];
            const int   n  = fx.taps[x];
            const float * wrow = fx.weights.data() + (size_t)x * fx.max_taps;
            uint8_t * out = dst + ((size_t)y * dst_w + x) * channels;
            float acc[8] = {0}; // up to 8 channels supported here
            for (int t = 0; t < n; ++t) {
                int sx = k0 + t;
                if (sx < 0)      sx = 0;
                if (sx >= src_w) sx = src_w - 1;
                const float w = wrow[t];
                const float * sp = tmp.data() + ((size_t)y * src_w + sx) * channels;
                for (int c = 0; c < channels; ++c) {
                    acc[c] += w * sp[c];
                }
            }
            for (int c = 0; c < channels; ++c) {
                int iv = (int)std::lrintf(acc[c]);
                if (iv < 0)   iv = 0;
                if (iv > 255) iv = 255;
                out[c] = (uint8_t)iv;
            }
        }
    }
}

// (target_h, target_w, 3) uint8 RGB -> (n_h*n_w, 3*p*p) fp32.
//
// Mirrors HF convert_image_to_patches (which gets a CHW tensor):
//   image (3, H, W) -> reshape (3, n_h, p, n_w, p) -> permute (1,3,2,4,0) -> reshape (n_h*n_w, p*p*3)
// Our input is HWC uint8; the channel-last patch layout expected by the encoder is the same
// as HF's reshape result because the original CHW layout has channels as the slowest dim, and
// after permute (1,3,2,4,0) channels become the FASTEST dim — matching HWC.
void rescale_normalize_patchify(
    const uint8_t * rgb,
    int             height,
    int             width,
    int             patch_size,
    float           rescale_factor,
    const float     mean[3],
    const float     std_v[3],
    std::vector<float> & out) {
    const int n_h = height / patch_size;
    const int n_w = width  / patch_size;
    const int feat = 3 * patch_size * patch_size;
    out.assign((size_t)n_h * n_w * feat, 0.0f);

    for (int ph = 0; ph < n_h; ++ph) {
        for (int pw = 0; pw < n_w; ++pw) {
            const size_t patch_idx = (size_t)ph * n_w + pw;
            float * dst = out.data() + patch_idx * feat;

            for (int py = 0; py < patch_size; ++py) {
                const int y = ph * patch_size + py;
                for (int px = 0; px < patch_size; ++px) {
                    const int x = pw * patch_size + px;
                    const uint8_t * src_pix = rgb + (y * width + x) * 3;
                    const size_t off = ((size_t)py * patch_size + px) * 3;
                    for (int c = 0; c < 3; ++c) {
                        float v = (float)src_pix[c] * rescale_factor;
                        v = (v - mean[c]) / std_v[c];
                        dst[off + c] = v;
                    }
                }
            }
        }
    }
}

} // anonymous namespace

bool preprocess_image_rgb(
    const uint8_t * rgb,
    int             height,
    int             width,
    int             max_num_patches,
    int             patch_size,
    float           rescale_factor,
    float           image_mean[3],
    float           image_std[3],
    PreprocResult & out,
    std::string &   error) {
    if (!rgb || height <= 0 || width <= 0) {
        error = "empty input";
        return false;
    }
    if (patch_size <= 0 || max_num_patches <= 0) {
        error = "invalid patch_size or max_num_patches";
        return false;
    }

    int target_h = 0, target_w = 0;
    compute_target_size(height, width, patch_size, max_num_patches, target_h, target_w);

    std::vector<uint8_t> resized((size_t)target_h * target_w * 3);
    resize_bilinear_u8(rgb, height, width, 3, resized.data(), target_h, target_w);

    rescale_normalize_patchify(
        resized.data(), target_h, target_w, patch_size,
        rescale_factor, image_mean, image_std,
        out.pixel_values);

    out.n_patches_h = target_h / patch_size;
    out.n_patches_w = target_w / patch_size;
    return true;
}

bool preprocess_image_file(
    const std::string & image_path,
    int                 max_num_patches,
    int                 patch_size,
    float               rescale_factor,
    float               image_mean[3],
    float               image_std[3],
    PreprocResult &     out,
    std::string &       error) {
    int w = 0, h = 0, ch = 0;
    uint8_t * pixels = stbi_load(image_path.c_str(), &w, &h, &ch, /*req_comp=*/3);
    if (!pixels) {
        error = std::string("stbi_load failed: ") + (stbi_failure_reason() ? stbi_failure_reason() : "?");
        return false;
    }
    bool ok = preprocess_image_rgb(
        pixels, h, w,
        max_num_patches, patch_size,
        rescale_factor, image_mean, image_std,
        out, error);
    stbi_image_free(pixels);
    return ok;
}

} // namespace siglip2
