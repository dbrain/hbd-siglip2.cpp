#include "siglip2_tokenizer.h"

#include "sentencepiece_processor.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace siglip2 {

struct Tokenizer::State {
    sentencepiece::SentencePieceProcessor proc;
};

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() {
    close();
}

void Tokenizer::close() {
    if (state_) {
        delete state_;
        state_ = nullptr;
    }
}

bool Tokenizer::load(const std::string & spm_model_path) {
    close();
    state_ = new State{};
    auto status = state_->proc.Load(spm_model_path);
    if (!status.ok()) {
        error_msg_ = std::string("sentencepiece Load failed: ") + status.ToString();
        delete state_;
        state_ = nullptr;
        return false;
    }
    pad_id_ = state_->proc.pad_id();
    bos_id_ = state_->proc.bos_id();
    eos_id_ = state_->proc.eos_id();
    if (pad_id_ < 0) {
        // Some sentencepiece models don't define pad_id (returns -1). For
        // SigLIP2/Gemma the convention is pad_token_id=0; fall back to that.
        pad_id_ = 0;
    }
    return true;
}

int Tokenizer::vocab_size() const {
    return state_ ? state_->proc.GetPieceSize() : 0;
}

bool Tokenizer::encode(
    const std::string &      text,
    int                      max_length,
    std::vector<int32_t> &   out_token_ids,
    std::vector<int32_t> &   out_attention_mask) {
    if (!state_) {
        error_msg_ = "Tokenizer not loaded";
        return false;
    }
    if (max_length <= 0) {
        error_msg_ = "max_length must be > 0";
        return false;
    }

    std::vector<int> ids;
    auto status = state_->proc.Encode(text, &ids);
    if (!status.ok()) {
        error_msg_ = std::string("Encode failed: ") + status.ToString();
        return false;
    }

    // SigLIP2's HF tokenizer appends EOS before padding (Gemma convention).
    // Sentencepiece's plain Encode doesn't, so we add it here. If truncation
    // would drop EOS, we keep it as the last position to mirror HF.
    if (eos_id_ >= 0) {
        ids.push_back(eos_id_);
    }

    out_token_ids.assign((size_t)max_length, pad_id_);
    out_attention_mask.assign((size_t)max_length, 0);

    const int n_keep = std::min((int)ids.size(), max_length);
    for (int i = 0; i < n_keep; ++i) {
        out_token_ids[i]     = ids[i];
        out_attention_mask[i] = 1;
    }
    // If we had to truncate and EOS got dropped, replace the last token with EOS.
    if (eos_id_ >= 0 && (int)ids.size() > max_length) {
        out_token_ids[max_length - 1] = eos_id_;
    }
    return true;
}

} // namespace siglip2
