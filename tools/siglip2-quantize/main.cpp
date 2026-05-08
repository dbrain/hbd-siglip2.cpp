// siglip2-quantize — re-quantize a siglip2 GGUF (typically the F16 produced by
// scripts/convert_siglip2_to_gguf.py --type f16) into a Q4_K_M GGUF, with
// K-padding on the innermost dim where required by the K-quant block size.
//
// We need a C++ tool because the Python `gguf` library raises NotImplementedError
// on K-quants — it only supports F16/F32/Q8_0/Q4_0 quantize. The C++ side
// uses ggml's `ggml_quantize_chunk`, which is the same code path llama.cpp's
// quantize binary uses.
//
// Per-tensor policy (siglip2-specific):
//   - F32 in input  → F32 in output (preserves biases, LN weights, position
//                     embeddings, probe head, logit_scale/bias).
//   - F16 in input  → Q4_K with K-padding to nearest 256 if not aligned. If
//                     the resulting type can't quantize (e.g., < 1 super-block
//                     row), fall back to F16 unchanged.
//
// K-padding zero-extends the innermost (== K, == in_features) dim with zeros.
// The siglip2 runtime is responsible for matching activation shapes (either
// via ggml_pad insertion at each padded mul_mat or by baking K-padding into
// the megakernel A0/A2/A3 kernels that own the upstream buffers).

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int Q4_K_BLOCK_SIZE = 256;

// Decide which output type to use for a given input tensor. Returns
// GGML_TYPE_COUNT to mean "copy as-is, no conversion".
//
// Mirrors llama.cpp's Q4_K_M recipe:
//   - Most 2D weights → Q4_K.
//   - ffn_down (== feed_forward.w2) → Q6_K (more bits, less quant noise on
//     the residual-affecting weight).
//   - attn_v ... siglip2 fuses Q/K/V into attn_qkv so we can't bump just V.
//     The whole fused qkv stays at Q4_K.
//
// For Q5_K_M, same idea but with Q5_K base + Q6_K ffn_down.
ggml_type decide_target_type(const ggml_tensor * t, ggml_type wanted_qtype) {
    // Preserve F32: biases, LN weights, position embeddings, probe head,
    // logit scale/bias — they're already F32 in the input GGUF and should
    // stay that way.
    if (t->type == GGML_TYPE_F32) {
        return GGML_TYPE_F32;
    }

    // 1D tensors (typically biases, LN weights cast to F16 — but our policy
    // already keeps them F32) shouldn't be quantized.
    if (ggml_n_dims(t) < 2) {
        return t->type;
    }

    const std::string name = ggml_get_name(t);

    // Q4_K_M / Q5_K_M recipe: ffn_down at Q6_K. Per Daniel Han's analysis
    // (and the original llama.cpp Q4_K_M decision), ffn_down absorbs the
    // residual-stream signal and is more sensitive to quant noise than
    // ffn_up or attn projections.
    if (wanted_qtype == GGML_TYPE_Q4_K || wanted_qtype == GGML_TYPE_Q5_K) {
        if (name.find("ffn_down.weight") != std::string::npos) {
            return GGML_TYPE_Q6_K;
        }
        // token_embd is read via ggml_get_rows. ggml-cuda only ships a Q4_K
        // get_rows kernel (per dbrain/ggml@4eec5550) — Q5_K/Q6_K get_rows
        // would crash. Pin token_embd to Q4_K regardless of base quant. The
        // VRAM hit is small (256k×1152 × 4.5 bits/8 = ~166 MiB; Q5_K would
        // be ~203 MiB).
        if (name == "t.token_embd.weight") {
            return GGML_TYPE_Q4_K;
        }
    }

    // For the rest (F16 weights), apply the wanted quant type.
    return wanted_qtype;
}

// Pad an F32 row buffer's innermost dim from k_orig to k_padded with zeros.
// Returns a new row-major buffer of shape (n_rows, k_padded).
std::vector<float> pad_innermost_f32(
        const float * src, int64_t n_rows, int64_t k_orig, int64_t k_padded) {
    std::vector<float> out((size_t) n_rows * (size_t) k_padded, 0.0f);
    for (int64_t r = 0; r < n_rows; ++r) {
        std::memcpy(out.data() + r * k_padded, src + r * k_orig,
                    (size_t) k_orig * sizeof(float));
        // Tail [k_orig, k_padded) already zero from the constructor.
    }
    return out;
}

// Convert F16 → F32 (full tensor).
std::vector<float> f16_to_f32(const ggml_fp16_t * src, size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(src[i]);
    return out;
}

// Quantize an F32 tensor (n_rows × k cols) to ggml_quantize_chunk's expected
// row layout. Returns the byte buffer.
std::vector<uint8_t> quantize_rows(
        ggml_type qtype, const float * src,
        int64_t n_rows, int64_t k) {
    const size_t row_bytes = ggml_row_size(qtype, k);
    std::vector<uint8_t> out(row_bytes * (size_t) n_rows);
    ggml_quantize_chunk(qtype, src, out.data(), 0, n_rows, k, nullptr);
    return out;
}

bool quantize_file(const std::string & in_path,
                   const std::string & out_path,
                   ggml_type           wanted_qtype,
                   ggml_ftype          target_ftype) {
    fprintf(stderr, "siglip2-quantize: loading '%s'\n", in_path.c_str());

    // Load input with allocation so t->data is populated for each tensor.
    ggml_context * ctx_in_ggml = nullptr;
    gguf_init_params params = { /*no_alloc=*/false, /*ctx=*/&ctx_in_ggml };
    gguf_context * ctx_in = gguf_init_from_file(in_path.c_str(), params);
    if (!ctx_in || !ctx_in_ggml) {
        fprintf(stderr, "siglip2-quantize: failed to open '%s'\n", in_path.c_str());
        return false;
    }

    // Build output gguf_context. Copy KV from input; bump file_type.
    gguf_context * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_in);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", (uint32_t) target_ftype);

    // We need a separate ggml_context for our OUTPUT meta-tensors (the
    // ones with K-padded shape and new dtypes). no_alloc since we just
    // need shape/type metadata; the data we manage out-of-band.
    const size_t n_tensors = (size_t) gguf_get_n_tensors(ctx_in);
    const size_t mem_size  = ggml_tensor_overhead() * (n_tensors + 16);
    std::vector<uint8_t> meta_mem(mem_size);
    ggml_init_params meta_params = { mem_size, meta_mem.data(), /*no_alloc=*/true };
    ggml_context * ctx_meta_out = ggml_init(meta_params);
    if (!ctx_meta_out) {
        fprintf(stderr, "siglip2-quantize: ggml_init for meta failed\n");
        return false;
    }

    // Storage for converted blobs — we keep them alive until the file is
    // written. Each entry holds the bytes for one tensor.
    std::vector<std::vector<uint8_t>> data_storage(n_tensors);

    int n_quantized = 0, n_kept = 0, n_kpad = 0;
    size_t in_bytes = 0, out_bytes = 0;

    for (int i = 0; i < (int) n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_in, i);
        ggml_tensor * t   = ggml_get_tensor(ctx_in_ggml, name);
        if (!t) {
            fprintf(stderr, "siglip2-quantize: missing tensor '%s' (skipping)\n", name);
            continue;
        }

        const int n_dims = ggml_n_dims(t);
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (int d = 0; d < n_dims; ++d) ne[d] = t->ne[d];
        const size_t in_size = ggml_nbytes(t);
        in_bytes += in_size;

        const ggml_type src_type = t->type;
        ggml_type dst_type = decide_target_type(t, wanted_qtype);

        // K-padding: only applies when the chosen dst_type is a quant whose
        // block size doesn't divide ne[0]. If padding fails (e.g. dst_type
        // doesn't support that K, or the quantize fails), fall back to F16.
        int64_t k_orig   = ne[0];
        int64_t k_padded = ne[0];
        bool needs_pad = false;
        if (ggml_is_quantized(dst_type)) {
            const int64_t blck = ggml_blck_size(dst_type);
            if (k_orig % blck != 0) {
                k_padded  = ((k_orig + blck - 1) / blck) * blck;
                needs_pad = (k_padded != k_orig);
            }
        }

        // Output shape: same as input, except ne[0] possibly padded.
        int64_t ne_out[GGML_MAX_DIMS];
        for (int d = 0; d < GGML_MAX_DIMS; ++d) ne_out[d] = ne[d];
        ne_out[0] = k_padded;

        ggml_tensor * t_out = ggml_new_tensor(
            ctx_meta_out, dst_type, n_dims, ne_out);
        if (!t_out) {
            fprintf(stderr, "siglip2-quantize: ggml_new_tensor failed for '%s'\n", name);
            return false;
        }
        ggml_set_name(t_out, name);

        // Compute the data blob.
        std::vector<uint8_t> blob;
        if (dst_type == src_type && !needs_pad) {
            // Pure copy (F32 stays F32, or no-op cases).
            blob.assign((const uint8_t *) t->data,
                        (const uint8_t *) t->data + in_size);
            ++n_kept;
        } else if (ggml_is_quantized(dst_type)) {
            // Convert to F32, K-pad if needed, quantize.
            int64_t n_rows = 1;
            for (int d = 1; d < n_dims; ++d) n_rows *= ne[d];

            std::vector<float> f32;
            if (src_type == GGML_TYPE_F16) {
                f32 = f16_to_f32((const ggml_fp16_t *) t->data,
                                 (size_t) n_rows * (size_t) k_orig);
            } else if (src_type == GGML_TYPE_F32) {
                f32.assign((const float *) t->data,
                           (const float *) t->data + (size_t) n_rows * (size_t) k_orig);
            } else {
                fprintf(stderr,
                    "siglip2-quantize: tensor '%s' has unsupported source type %s; "
                    "copying unchanged\n", name, ggml_type_name(src_type));
                blob.assign((const uint8_t *) t->data,
                            (const uint8_t *) t->data + in_size);
                // Reset t_out to the original type/shape since we're not
                // converting.
                // (We can't easily un-add from ctx_meta_out, but
                // gguf_add_tensor below uses the ggml_tensor's metadata —
                // so let's NOT add this t_out and recreate.)
                // Rebuild t_out with original type/shape.
                // ...simplest: reset ne_out to ne and dst_type to src_type.
                t_out->type = src_type;
                for (int d = 0; d < GGML_MAX_DIMS; ++d) t_out->ne[d] = ne[d];
                ++n_kept;
                gguf_add_tensor(ctx_out, t_out);
                gguf_set_tensor_data(ctx_out, name, blob.data());
                data_storage[i] = std::move(blob);
                out_bytes += data_storage[i].size();
                fprintf(stderr, "  [%3d/%3zu] %-44s  copy %s (unsupported src)\n",
                        i + 1, n_tensors, name, ggml_type_name(src_type));
                continue;
            }

            std::vector<float> padded;
            const float * f32_for_quant;
            if (needs_pad) {
                padded = pad_innermost_f32(f32.data(), n_rows, k_orig, k_padded);
                f32_for_quant = padded.data();
                ++n_kpad;
            } else {
                f32_for_quant = f32.data();
            }

            blob = quantize_rows(dst_type, f32_for_quant, n_rows, k_padded);
            ++n_quantized;
        } else if (dst_type == GGML_TYPE_F16 && src_type == GGML_TYPE_F32) {
            // F32 → F16 (we don't normally hit this for siglip2, but safe).
            const size_t n = (size_t) ggml_nelements(t);
            std::vector<ggml_fp16_t> f16(n);
            for (size_t k = 0; k < n; ++k)
                f16[k] = ggml_fp32_to_fp16(((const float *) t->data)[k]);
            blob.assign((const uint8_t *) f16.data(),
                        (const uint8_t *) f16.data() + n * sizeof(ggml_fp16_t));
            ++n_quantized;
        } else {
            // Default: copy as-is.
            blob.assign((const uint8_t *) t->data,
                        (const uint8_t *) t->data + in_size);
            ++n_kept;
        }

        gguf_add_tensor(ctx_out, t_out);
        gguf_set_tensor_data(ctx_out, name, blob.data());
        data_storage[i] = std::move(blob);
        out_bytes += data_storage[i].size();

        fprintf(stderr, "  [%3d/%3zu] %-44s  %s -> %s",
                i + 1, n_tensors, name,
                ggml_type_name(src_type), ggml_type_name(dst_type));
        if (needs_pad) {
            fprintf(stderr, "  K %lld -> %lld",
                    (long long) k_orig, (long long) k_padded);
        }
        fprintf(stderr, "\n");
    }

    fprintf(stderr,
        "siglip2-quantize: %d quantized, %d kept, %d K-padded.  "
        "in=%.1f MB  out=%.1f MB\n",
        n_quantized, n_kept, n_kpad,
        in_bytes / (1024.0 * 1024.0),
        out_bytes / (1024.0 * 1024.0));

    fprintf(stderr, "siglip2-quantize: writing '%s'\n", out_path.c_str());
    gguf_write_to_file(ctx_out, out_path.c_str(), /*only_meta=*/false);

    gguf_free(ctx_in);
    gguf_free(ctx_out);
    ggml_free(ctx_meta_out);
    ggml_free(ctx_in_ggml);
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr,
            "usage: %s in.gguf out.gguf <q4_k_m|q4_k|q8_0|q4_0>\n",
            argv[0]);
        return 1;
    }
    const std::string in_path  = argv[1];
    const std::string out_path = argv[2];
    const std::string qtype_arg = argv[3];

    ggml_type  wanted_qtype  = GGML_TYPE_COUNT;
    ggml_ftype target_ftype  = GGML_FTYPE_UNKNOWN;
    if (qtype_arg == "q4_k_m" || qtype_arg == "q4_k") {
        wanted_qtype = GGML_TYPE_Q4_K;
        target_ftype = GGML_FTYPE_MOSTLY_Q4_K;
    } else if (qtype_arg == "q5_k_m" || qtype_arg == "q5_k") {
        wanted_qtype = GGML_TYPE_Q5_K;
        target_ftype = GGML_FTYPE_MOSTLY_Q5_K;
    } else if (qtype_arg == "q6_k") {
        wanted_qtype = GGML_TYPE_Q6_K;
        target_ftype = GGML_FTYPE_MOSTLY_Q6_K;
    } else if (qtype_arg == "q8_0") {
        wanted_qtype = GGML_TYPE_Q8_0;
        target_ftype = GGML_FTYPE_MOSTLY_Q8_0;
    } else if (qtype_arg == "q4_0") {
        wanted_qtype = GGML_TYPE_Q4_0;
        target_ftype = GGML_FTYPE_MOSTLY_Q4_0;
    } else {
        fprintf(stderr, "siglip2-quantize: unknown qtype '%s'\n", qtype_arg.c_str());
        return 1;
    }

    if (!quantize_file(in_path, out_path, wanted_qtype, target_ftype)) {
        return 1;
    }
    return 0;
}
