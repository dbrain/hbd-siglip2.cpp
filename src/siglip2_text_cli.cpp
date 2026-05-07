#include "siglip2_text.h"
#include "siglip2_tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char * argv0) {
    fprintf(stderr,
        "siglip2-text-cli (text encoder)\n"
        "\n"
        "Usage:\n"
        "  %s --model <gguf> --tokenizer <spm.model> --text \"prompt\" [--out file]\n"
        "  %s --model <gguf> --token-ids tok.bin --attention-mask mask.bin  [--out file]\n"
        "\n"
        "  --model           SigLIP2 GGUF (must have t.* tensors)\n"
        "  --tokenizer       sentencepiece tokenizer.model file (Gemma 256K)\n"
        "  --text            free-form text input (tokenize via --tokenizer)\n"
        "  --token-ids       OR: pre-tokenized int32 buffer of length max_position_embeddings\n"
        "  --attention-mask  same length (1=valid, 0=pad); required with --token-ids\n"
        "  --out             write fp32 embedding to this file (default: stdout text)\n",
        argv0, argv0);
}

bool read_bytes(const std::string & p, std::vector<uint8_t> & out, std::string & err) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { err = "open: " + p; return false; }
    auto sz = f.tellg();
    f.seekg(0);
    out.resize((size_t)sz);
    f.read(reinterpret_cast<char *>(out.data()), sz);
    return f.good();
}

bool write_bytes(const std::string & p, const void * data, size_t n, std::string & err) {
    std::ofstream f(p, std::ios::binary);
    if (!f) { err = "write: " + p; return false; }
    f.write(reinterpret_cast<const char *>(data), (std::streamsize)n);
    return f.good();
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path, tok_path, mask_path, out_path;
    std::string tokenizer_path, text_input;
    for (int i = 1; i < argc; ++i) {
        if      ((!strcmp(argv[i], "--model") || !strcmp(argv[i], "-m")) && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc)        tokenizer_path = argv[++i];
        else if (!strcmp(argv[i], "--text") && i + 1 < argc)             text_input = argv[++i];
        else if (!strcmp(argv[i], "--token-ids") && i + 1 < argc)        tok_path = argv[++i];
        else if (!strcmp(argv[i], "--attention-mask") && i + 1 < argc)   mask_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)              out_path = argv[++i];
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); print_usage(argv[0]); return 2; }
    }
    if (model_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }
    const bool use_text  = !tokenizer_path.empty() && !text_input.empty();
    const bool use_bin   = !tok_path.empty() && !mask_path.empty();
    if (use_text == use_bin) {
        fprintf(stderr, "Provide exactly one of (--tokenizer + --text) or (--token-ids + --attention-mask)\n");
        return 2;
    }

    siglip2::TextEncoder enc;
    if (!enc.load(model_path)) {
        fprintf(stderr, "load failed: %s\n", enc.last_error().c_str());
        return 1;
    }
    const int seq = enc.config().max_position_embeddings;

    std::vector<int32_t> token_ids, attention_mask;
    std::string err;

    if (use_text) {
        siglip2::Tokenizer tk;
        if (!tk.load(tokenizer_path)) {
            fprintf(stderr, "tokenizer load failed: %s\n", tk.last_error().c_str());
            return 1;
        }
        if (!tk.encode(text_input, seq, token_ids, attention_mask)) {
            fprintf(stderr, "tokenize failed: %s\n", tk.last_error().c_str());
            return 1;
        }
        fprintf(stderr, "tokenized: pad=%d vocab=%d active=%d/%d\n",
            tk.pad_token_id(), tk.vocab_size(),
            (int)std::count(attention_mask.begin(), attention_mask.end(), 1), seq);
    } else {
        std::vector<uint8_t> tok_raw, mask_raw;
        if (!read_bytes(tok_path, tok_raw, err))  { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        if (!read_bytes(mask_path, mask_raw, err)){ fprintf(stderr, "%s\n", err.c_str()); return 1; }
        const size_t expected = sizeof(int32_t) * (size_t)seq;
        if (tok_raw.size() != expected || mask_raw.size() != expected) {
            fprintf(stderr,
                "size mismatch: tok=%zu mask=%zu expected=%zu (seq=%d sizeof(int32)=%zu)\n",
                tok_raw.size(), mask_raw.size(), expected, seq, sizeof(int32_t));
            return 1;
        }
        token_ids.assign(reinterpret_cast<const int32_t *>(tok_raw.data()),
                         reinterpret_cast<const int32_t *>(tok_raw.data()) + seq);
        attention_mask.assign(reinterpret_cast<const int32_t *>(mask_raw.data()),
                              reinterpret_cast<const int32_t *>(mask_raw.data()) + seq);
    }

    std::vector<float> emb;
    if (!enc.encode(token_ids.data(), attention_mask.data(), emb)) {
        fprintf(stderr, "encode failed: %s\n", enc.last_error().c_str());
        return 1;
    }

    if (!out_path.empty()) {
        if (!write_bytes(out_path, emb.data(), sizeof(float) * emb.size(), err)) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        fprintf(stderr, "wrote text embedding (%zu floats) to %s\n", emb.size(), out_path.c_str());
    } else {
        fprintf(stdout, "text_embed[%zu]:\n", emb.size());
        for (size_t i = 0; i < std::min<size_t>(emb.size(), 8); ++i) {
            fprintf(stdout, "  [%zu]=%g\n", i, emb[i]);
        }
    }
    return 0;
}
