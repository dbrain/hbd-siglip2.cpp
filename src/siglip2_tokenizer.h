#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace siglip2 {

// Multilingual sentencepiece tokenizer for SigLIP2 (Gemma-style 256K vocab).
// Loads `tokenizer.model` (sentencepiece proto). Padding mode mirrors HF:
// padding="max_length", truncation, return_attention_mask.
class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    bool load(const std::string & spm_model_path);
    void close();

    int  pad_token_id() const { return pad_id_; }
    int  bos_token_id() const { return bos_id_; }
    int  eos_token_id() const { return eos_id_; }
    int  vocab_size()   const;
    const std::string & last_error() const { return error_msg_; }

    // Encode a single text. Pads / truncates to `max_length`. Emits two
    // arrays of length `max_length` (token_ids and attention_mask, where
    // attention_mask[i]=1 iff position i is a non-pad token).
    bool encode(
        const std::string &      text,
        int                      max_length,
        std::vector<int32_t> &   out_token_ids,
        std::vector<int32_t> &   out_attention_mask);

private:
    struct State;
    State *      state_ = nullptr;
    std::string  error_msg_;
    int          pad_id_ = 0;
    int          bos_id_ = -1;
    int          eos_id_ = -1;
};

} // namespace siglip2
