#include "siglip2_vision.h"

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
        "siglip2-cli (M1 vision-only)\n"
        "\n"
        "Usage: %s --model <path.gguf> --pixel-values <path.bin> [--n-patches N] [--out <path.bin>]\n"
        "\n"
        "  --model         GGUF file produced by scripts/convert_siglip2_to_gguf.py\n"
        "  --pixel-values  raw fp32 buffer of shape [n_patches, num_channels*patch_size^2]\n"
        "                  (preprocess via HF Siglip2ImageProcessor; pickle-free for parity tests)\n"
        "  --n-patches     number of patches in the buffer; defaults to GGUF native (256)\n"
        "  --out           write embedding to this file as raw fp32; default: stdout text\n"
        "\n"
        "Stdout (when --out omitted):\n"
        "  embedding[H]:\n"
        "    [0]=...\n"
        "    ...\n",
        argv0);
}

bool read_file(const std::string & path, std::vector<uint8_t> & out, std::string & err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        err = "could not open: " + path;
        return false;
    }
    auto sz = f.tellg();
    f.seekg(0);
    out.resize((size_t)sz);
    f.read(reinterpret_cast<char *>(out.data()), sz);
    return f.good();
}

bool write_file(const std::string & path, const void * data, size_t bytes, std::string & err) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        err = "could not write: " + path;
        return false;
    }
    f.write(reinterpret_cast<const char *>(data), (std::streamsize)bytes);
    return f.good();
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string pixel_path;
    std::string out_path;
    int         n_patches = -1; // default: use config.num_patches

    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "--model") || !strcmp(argv[i], "-m")) && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "--pixel-values") && i + 1 < argc) {
            pixel_path = argv[++i];
        } else if (!strcmp(argv[i], "--n-patches") && i + 1 < argc) {
            n_patches = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }

    if (model_path.empty() || pixel_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    siglip2::VisionEncoder enc;
    if (!enc.load(model_path)) {
        fprintf(stderr, "load failed: %s\n", enc.last_error().c_str());
        return 1;
    }

    const auto & cfg = enc.config();
    if (n_patches < 0) n_patches = cfg.num_patches;
    const int feat = cfg.num_channels * cfg.patch_size * cfg.patch_size;

    std::vector<uint8_t> raw;
    std::string err;
    if (!read_file(pixel_path, raw, err)) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const size_t expected = sizeof(float) * (size_t)n_patches * feat;
    if (raw.size() != expected) {
        fprintf(stderr,
            "pixel buffer size mismatch: got %zu bytes, expected %zu (n_patches=%d feat=%d sizeof(float)=%zu)\n",
            raw.size(), expected, n_patches, feat, sizeof(float));
        return 1;
    }

    std::vector<float> emb;
    if (!enc.encode(reinterpret_cast<const float *>(raw.data()), n_patches, siglip2::Pooling::MEAN, emb)) {
        fprintf(stderr, "encode failed: %s\n", enc.last_error().c_str());
        return 1;
    }

    if (!out_path.empty()) {
        if (!write_file(out_path, emb.data(), sizeof(float) * emb.size(), err)) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        fprintf(stderr, "wrote embedding (%zu floats) to %s\n", emb.size(), out_path.c_str());
    } else {
        fprintf(stdout, "embedding[%zu]:\n", emb.size());
        for (size_t i = 0; i < std::min<size_t>(emb.size(), 8); ++i) {
            fprintf(stdout, "  [%zu]=%g\n", i, emb[i]);
        }
        // Plus a few from the tail so a sanity glance has more signal.
        if (emb.size() > 16) {
            fprintf(stdout, "  ...\n");
            for (size_t i = emb.size() - 4; i < emb.size(); ++i) {
                fprintf(stdout, "  [%zu]=%g\n", i, emb[i]);
            }
        }
    }
    return 0;
}
