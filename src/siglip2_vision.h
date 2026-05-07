#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace siglip2 {

// Pooling strategy applied to encoder hidden states to produce the output
// embedding. Mirrors kobbler-vision's HTTP `pooling` parameter; M1 ships MEAN
// only, M2 adds PROBE.
enum class Pooling {
    MEAN,    // mean over patches (matches HF last_hidden_state.mean(dim=1))
    PROBE,   // Siglip2MultiheadAttentionPoolingHead (M2)
};

struct VisionConfig {
    int   hidden_size         = 1152;
    int   intermediate_size   = 4304;
    int   num_attention_heads = 16;
    int   num_hidden_layers   = 27;
    int   patch_size          = 16;
    int   num_patches         = 256; // native 16x16 grid
    int   num_channels        = 3;
    float layer_norm_eps      = 1e-6f;
};

// SigLIP2 vision encoder.
//
// M1: fixed-resolution path. Caller must pre-patchify the image to exactly
// `n_patches = num_patches` rows of `num_channels * patch_size^2` fp32 features.
// Pooling fixed to MEAN for M1.
//
// M2 will add NaFlex variable-resolution + bilinear pos-embed interp,
// attention masks for padded patches, and the PROBE pooling head.
class VisionEncoder {
public:
    VisionEncoder();
    ~VisionEncoder();

    bool load(const std::string & gguf_path);
    void close();

    const VisionConfig & config() const { return config_; }
    const std::string  & last_error() const { return error_msg_; }

    // pixel_values: contiguous fp32 of shape [n_patches, num_channels*patch_size^2]
    // out_embedding: resized to hidden_size on success.
    bool encode(
        const float *        pixel_values,
        int                  n_patches,
        Pooling              pooling,
        std::vector<float> & out_embedding);

private:
    VisionConfig config_;
    std::string  error_msg_;
    struct State;
    State * state_ = nullptr;
};

} // namespace siglip2
