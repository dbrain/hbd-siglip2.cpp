#include "siglip2_score.h"

#include "gguf_loader.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace siglip2 {

bool read_score_params(
    const std::string & gguf_path,
    ScoreParams &       out,
    std::string &       error) {
    qwen3_tts::GGUFLoader loader;
    if (!loader.open(gguf_path)) {
        error = loader.get_error();
        return false;
    }
    // mm.logit_scale and mm.logit_bias are stored as [1]-shaped F32 tensors.
    // Reading via the gguf data offset + tensor info is more painful than just
    // pulling them via ggml_backend_tensor_get; for now we read the raw bytes.
    gguf_context * ctx  = loader.get_ctx();
    ggml_context * meta = loader.get_meta_ctx();

    auto load_scalar = [&](const char * name, float & dst) -> bool {
        ggml_tensor * t = ggml_get_tensor(meta, name);
        if (!t) {
            error = std::string("Missing tensor: ") + name;
            return false;
        }
        // Find the tensor index, get its file offset, read 4 bytes.
        const int64_t n = gguf_get_n_tensors(ctx);
        int64_t idx = -1;
        for (int64_t i = 0; i < n; ++i) {
            if (std::string(gguf_get_tensor_name(ctx, i)) == name) {
                idx = i;
                break;
            }
        }
        if (idx < 0) {
            error = std::string("Tensor not found in gguf: ") + name;
            return false;
        }
        const size_t data_offset = gguf_get_data_offset(ctx);
        const size_t off         = gguf_get_tensor_offset(ctx, idx);
        FILE * f = fopen(gguf_path.c_str(), "rb");
        if (!f) {
            error = "open " + gguf_path;
            return false;
        }
        if (fseek(f, (long)(data_offset + off), SEEK_SET) != 0) {
            fclose(f);
            error = "seek failed";
            return false;
        }
        if (fread(&dst, sizeof(float), 1, f) != 1) {
            fclose(f);
            error = "read failed";
            return false;
        }
        fclose(f);
        return true;
    };

    if (!load_scalar("mm.logit_scale", out.logit_scale)) return false;
    if (!load_scalar("mm.logit_bias",  out.logit_bias))  return false;
    return true;
}

namespace {

void l2_normalize_rows(float * x, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float * row = x + (size_t)r * cols;
        double sumsq = 0.0;
        for (int c = 0; c < cols; ++c) sumsq += (double)row[c] * (double)row[c];
        const double norm = std::sqrt(sumsq) + 1e-12;
        const float  inv  = (float)(1.0 / norm);
        for (int c = 0; c < cols; ++c) row[c] *= inv;
    }
}

} // namespace

void score_image_text(
    const float *   image_embeds, int n_image,
    const float *   text_embeds,  int n_text,
    int             hidden,
    const ScoreParams & params,
    float *         logits_per_image,
    float *         probs_per_image) {
    // Copy + normalize.
    std::vector<float> img((size_t)n_image * hidden);
    std::vector<float> txt((size_t)n_text  * hidden);
    std::memcpy(img.data(), image_embeds, sizeof(float) * img.size());
    std::memcpy(txt.data(), text_embeds,  sizeof(float) * txt.size());
    l2_normalize_rows(img.data(), n_image, hidden);
    l2_normalize_rows(txt.data(), n_text,  hidden);

    const float scale = std::exp(params.logit_scale);
    const float bias  = params.logit_bias;

    // logits_per_image[i, j] = (img_n[i] . txt_n[j]) * scale + bias
    for (int i = 0; i < n_image; ++i) {
        const float * imrow = img.data() + (size_t)i * hidden;
        for (int j = 0; j < n_text; ++j) {
            const float * txrow = txt.data() + (size_t)j * hidden;
            double dot = 0.0;
            for (int c = 0; c < hidden; ++c) dot += (double)imrow[c] * (double)txrow[c];
            const float logit = (float)(dot * (double)scale + (double)bias);
            if (logits_per_image) {
                logits_per_image[(size_t)i * n_text + j] = logit;
            }
            if (probs_per_image) {
                // Sigmoid, numerically stable.
                float p;
                if (logit >= 0.0f) {
                    const float e = std::exp(-logit);
                    p = 1.0f / (1.0f + e);
                } else {
                    const float e = std::exp(logit);
                    p = e / (1.0f + e);
                }
                probs_per_image[(size_t)i * n_text + j] = p;
            }
        }
    }
}

} // namespace siglip2
