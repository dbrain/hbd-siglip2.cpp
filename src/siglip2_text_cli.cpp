#include "siglip2_text.h"

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
        "siglip2-text-cli (text encoder, M3 parity tool)\n"
        "\n"
        "Usage: %s --model <path.gguf> --token-ids <path.bin> --attention-mask <path.bin> [--out <path.bin>]\n"
        "\n"
        "  --model           SigLIP2 GGUF (must have t.* tensors)\n"
        "  --token-ids       int32 buffer of length max_position_embeddings\n"
        "  --attention-mask  int32 buffer same length (1=valid, 0=pad)\n"
        "  --out             write fp32 embedding to this file (default: stdout text)\n",
        argv0);
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
    for (int i = 1; i < argc; ++i) {
        if      ((!strcmp(argv[i], "--model") || !strcmp(argv[i], "-m")) && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--token-ids") && i + 1 < argc)        tok_path = argv[++i];
        else if (!strcmp(argv[i], "--attention-mask") && i + 1 < argc)   mask_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)              out_path = argv[++i];
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); print_usage(argv[0]); return 2; }
    }
    if (model_path.empty() || tok_path.empty() || mask_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    siglip2::TextEncoder enc;
    if (!enc.load(model_path)) {
        fprintf(stderr, "load failed: %s\n", enc.last_error().c_str());
        return 1;
    }
    const int seq = enc.config().max_position_embeddings;

    std::string err;
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

    std::vector<float> emb;
    if (!enc.encode(reinterpret_cast<const int32_t *>(tok_raw.data()),
                    reinterpret_cast<const int32_t *>(mask_raw.data()),
                    emb)) {
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
