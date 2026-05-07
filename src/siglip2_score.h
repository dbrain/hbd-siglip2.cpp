#pragma once

#include <string>
#include <vector>

namespace siglip2 {

// Sigmoid image-text scoring (Siglip2Model contrastive head).
//
// HF math:
//   image_embed_n = image_embed / ||image_embed||
//   text_embed_n  = text_embed  / ||text_embed||
//   logits_per_text  = text_n @ image_n.T * exp(logit_scale) + logit_bias  // (n_txt, n_img)
//   logits_per_image = logits_per_text.T                                    // (n_img, n_txt)
//   probs            = sigmoid(logits)
//
// `image_embeds` and `text_embeds` are flattened row-major buffers of size
// (n_*, hidden). Outputs `logits_per_image` and `probs_per_image` are
// (n_image, n_text) row-major.

struct ScoreParams {
    float logit_scale = 0.0f; // scalar, gets exponentiated
    float logit_bias  = 0.0f;
};

// Read mm.logit_scale and mm.logit_bias from a SigLIP2 GGUF file.
bool read_score_params(
    const std::string & gguf_path,
    ScoreParams &       out,
    std::string &       error);

// Compute logits_per_image (n_image, n_text) and the corresponding
// sigmoid probabilities. Either output may be nullptr to skip.
void score_image_text(
    const float *   image_embeds, int n_image,
    const float *   text_embeds,  int n_text,
    int             hidden,
    const ScoreParams & params,
    float *         logits_per_image,
    float *         probs_per_image);

} // namespace siglip2
