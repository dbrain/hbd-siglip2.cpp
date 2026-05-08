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

    // private_backend=true gives this encoder its own ggml_backend_t (and thus
    // its own CUDA stream), so concurrent calls into other encoders execute on
    // separate streams. Default is the process-wide shared backend.
    bool load(const std::string & gguf_path, bool private_backend = false);
    void close();

    const TextConfig & config() const { return config_; }
    const std::string & last_error() const { return error_msg_; }

    // token_ids:        length n_tokens (must be > 0 and <= max_position_embeddings)
    // attention_mask:   length n_tokens (1=valid, 0=pad). May be nullptr if no padding.
    // out_embedding:    resized to projection_size on success.
    //
    // Pools at position n_tokens-1 (the last position in the batch), matching HF's
    // last_hidden_state[:, -1, :]. For multi-prompt batches the caller pads all
    // prompts to the same n_tokens (longest-in-batch) for HF parity.
    bool encode(
        const int32_t *      token_ids,
        int                  n_tokens,
        const int32_t *      attention_mask, // may be nullptr
        std::vector<float> & out_embedding);

    // Batched encode: all n_batch prompts share the same n_tokens (caller pads
    // each to the same length, typically max_position_embeddings). token_ids
    // is a flat array of n_tokens*n_batch I32 values, batch-major. attention_mask
    // currently must be nullptr (matches HF — no mask used).
    //
    // out_embeddings is resized to projection_size*n_batch on success, batch-major
    // (batch i embedding starts at offset i*projection_size).
    //
    // Hits a single graph compute for all n_batch prompts — amortizes launch
    // overhead. For n_batch==1, dispatches to encode() to keep the megakernel
    // hooks firing on that path.
    bool encode_batch(
        const int32_t *      token_ids,
        int                  n_tokens,
        int                  n_batch,
        const int32_t *      attention_mask,
        std::vector<float> & out_embeddings);

private:
    TextConfig  config_;
    std::string error_msg_;
    struct State;
    State * state_ = nullptr;
};

} // namespace siglip2
