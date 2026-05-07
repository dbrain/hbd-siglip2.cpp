#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace siglip2 {

// Result of preprocessing an image for the SigLIP2 vision encoder.
//
// Layout matches the encoder's expected input: pixel_values is row-major
// fp32 with `n_patches_h * n_patches_w` rows of `num_channels * patch_size^2`
// floats. The encoder's NaFlex path snaps to exact grid (no padding), so this
// mirrors HF's pixel_values truncated to the active prefix.
struct PreprocResult {
    std::vector<float> pixel_values; // size = n_patches * 3 * patch_size * patch_size
    int                n_patches_h = 0;
    int                n_patches_w = 0;
};

// Preprocess an image file (any format stb_image handles: JPEG/PNG/BMP/...).
//
// Algorithm mirrors HF Siglip2ImageProcessor:
// 1. Load + convert to RGB uint8.
// 2. Binary-search the largest scale s.t. ceil(H*s/p)*ceil(W*s/p) <= max_num_patches,
//    rounding each dim up to a patch_size multiple.
// 3. Bilinear resize to (target_h, target_w).
// 4. Rescale: pixel * rescale_factor.
// 5. Normalize: (pixel - mean) / std.
// 6. Patchify: rearrange into (n_h*n_w, p*p*3) with channel-last memory order.
//
// On failure returns false and sets `error` to a message.
bool preprocess_image_file(
    const std::string & image_path,
    int                 max_num_patches,
    int                 patch_size,
    float               rescale_factor, // typically 1.0/255.0
    float               image_mean[3],  // typically {0.5, 0.5, 0.5}
    float               image_std[3],
    PreprocResult &     out,
    std::string &       error);

// Same, but starting from a pre-decoded RGB uint8 buffer (channels-last:
// data[y * width * 3 + x * 3 + c]).
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
    std::string &   error);

} // namespace siglip2
