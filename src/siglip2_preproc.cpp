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

// Bilinear resize for HxWxC uint8 -> dst_h x dst_w x C uint8.
// Matches PIL.Image.BILINEAR (no antialias). Channels-last memory order.
//
// PIL's bilinear maps each output pixel to a "centered" source coordinate:
// src = (out + 0.5) * src_size / dst_size - 0.5. This matches torchvision's
// interpolation when antialias=False.
void resize_bilinear_u8(
    const uint8_t * src,
    int             src_h,
    int             src_w,
    int             channels,
    uint8_t *       dst,
    int             dst_h,
    int             dst_w) {
    const double sy = (double)src_h / (double)dst_h;
    const double sx = (double)src_w / (double)dst_w;

    for (int y = 0; y < dst_h; ++y) {
        double fy = (y + 0.5) * sy - 0.5;
        int    y0 = (int)std::floor(fy);
        double dy = fy - y0;
        int    y1 = y0 + 1;
        if (y0 < 0)        { y0 = 0; dy = 0.0; }
        if (y0 >= src_h-1) { y0 = src_h - 1; y1 = y0; dy = 0.0; }
        else                { y1 = std::min(y1, src_h - 1); }

        for (int x = 0; x < dst_w; ++x) {
            double fx = (x + 0.5) * sx - 0.5;
            int    x0 = (int)std::floor(fx);
            double dx = fx - x0;
            int    x1 = x0 + 1;
            if (x0 < 0)        { x0 = 0; dx = 0.0; }
            if (x0 >= src_w-1) { x0 = src_w - 1; x1 = x0; dx = 0.0; }
            else                { x1 = std::min(x1, src_w - 1); }

            const uint8_t * p00 = src + (y0 * src_w + x0) * channels;
            const uint8_t * p01 = src + (y0 * src_w + x1) * channels;
            const uint8_t * p10 = src + (y1 * src_w + x0) * channels;
            const uint8_t * p11 = src + (y1 * src_w + x1) * channels;
            uint8_t *       po  = dst + (y  * dst_w + x ) * channels;

            const double w00 = (1.0 - dx) * (1.0 - dy);
            const double w01 = dx         * (1.0 - dy);
            const double w10 = (1.0 - dx) * dy;
            const double w11 = dx         * dy;

            for (int c = 0; c < channels; ++c) {
                double v = p00[c]*w00 + p01[c]*w01 + p10[c]*w10 + p11[c]*w11;
                int    iv = (int)std::round(v);
                if (iv < 0)   iv = 0;
                if (iv > 255) iv = 255;
                po[c] = (uint8_t)iv;
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
