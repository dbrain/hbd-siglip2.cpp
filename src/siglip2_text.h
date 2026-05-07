#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace siglip2 {

struct TextConfig {
    int   hidden_size             = 1152;
    int   intermediate_size       = 4304;
    int   num_attention_heads     = 16;
    int   num_hidden_layers       = 27;
    int   vocab_size              = 256000;
    int   max_position_embeddings = 64;
    int   projection_size         = 1152;
    float layer_norm_eps          = 1e-6f;
};

// SigLIP2 text encoder. Produces a projection_size-dim embedding from a fixed
// `max_position_embeddings`-length token sequence (HF uses padding="max_length").
//
// Pooling: last position then Linear head, matching HF Siglip2TextModel.
// Bidirectional self-attention with key-side padding mask (no causal mask).
class TextEncoder {
public:
    TextEncoder();
    ~TextEncoder();

    bool load(const std::string & gguf_path);
    void close();

    const TextConfig & config() const { return config_; }
    const std::string & last_error() const { return error_msg_; }

    // token_ids:        length = config.max_position_embeddings, int32
    // attention_mask:   length = config.max_position_embeddings, int32 (1=valid, 0=pad)
    // out_embedding:    resized to projection_size on success
    bool encode(
        const int32_t *      token_ids,
        const int32_t *      attention_mask,
        std::vector<float> & out_embedding);

private:
    TextConfig  config_;
    std::string error_msg_;
    struct State;
    State * state_ = nullptr;
};

} // namespace siglip2
