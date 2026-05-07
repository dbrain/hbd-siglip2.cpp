#include "siglip2_vision.h"
#include "siglip2_preproc.h"
#include "cuda/siglip2_megakernel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char * argv0) {
    fprintf(stderr,
        "siglip2-cli (vision encoder)\n"
        "\n"
        "Usage: %s --model <path.gguf> [--image <img>|--pixel-values <bin>] [opts]\n"
        "\n"
        "  --model            GGUF file from scripts/convert_siglip2_to_gguf.py\n"
        "  --image            image file (jpg/png/bmp/...) — preprocess in-process\n"
        "  --pixel-values     raw fp32 buffer of shape [n_patches, num_channels*patch_size^2]\n"
        "  --n-patches        with --pixel-values: number of patches in the buffer\n"
        "  --shape H,W        spatial patch grid (rows, cols); defaults to native square\n"
        "  --max-num-patches  with --image: cap on patches (HF default 256; kobbler 729)\n"
        "  --pooling          'probe' (default; matches HF pooler_output) or 'mean'\n"
        "  --out              write embedding to this file as raw fp32; default: stdout text\n",
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
    siglip2_megakernel::install();
    std::string model_path;
    std::string pixel_path;
    std::string image_path;
    std::string out_path;
    std::string pooling_str = "probe";
    int         n_patches = -1; // default: use config.num_patches
    int         n_h = -1;
    int         n_w = -1;
    int         max_num_patches = 256;

    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "--model") || !strcmp(argv[i], "-m")) && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "--pixel-values") && i + 1 < argc) {
            pixel_path = argv[++i];
        } else if (!strcmp(argv[i], "--image") && i + 1 < argc) {
            image_path = argv[++i];
        } else if (!strcmp(argv[i], "--max-num-patches") && i + 1 < argc) {
            max_num_patches = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--n-patches") && i + 1 < argc) {
            n_patches = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--shape") && i + 1 < argc) {
            // "--shape H,W" specifies non-square patch grids for naflex
            const char * s = argv[++i];
            char * comma = const_cast<char *>(strchr(s, ','));
            if (!comma) {
                fprintf(stderr, "--shape expects H,W (got: %s)\n", s);
                return 2;
            }
            *comma = '\0';
            n_h = std::atoi(s);
            n_w = std::atoi(comma + 1);
        } else if (!strcmp(argv[i], "--pooling") && i + 1 < argc) {
            pooling_str = argv[++i];
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

    siglip2::Pooling pooling;
    if (pooling_str == "mean") {
        pooling = siglip2::Pooling::MEAN;
    } else if (pooling_str == "probe") {
        pooling = siglip2::Pooling::PROBE;
    } else {
        fprintf(stderr, "Unknown --pooling: %s (use 'probe' or 'mean')\n", pooling_str.c_str());
        return 2;
    }

    if (model_path.empty() || (pixel_path.empty() && image_path.empty()) ||
        (!pixel_path.empty() && !image_path.empty())) {
        fprintf(stderr, "must supply --model and exactly one of --image / --pixel-values\n");
        print_usage(argv[0]);
        return 2;
    }

    siglip2::VisionEncoder enc;
    if (!enc.load(model_path)) {
        fprintf(stderr, "load failed: %s\n", enc.last_error().c_str());
        return 1;
    }

    const auto & cfg = enc.config();
    const int feat = cfg.num_channels * cfg.patch_size * cfg.patch_size;

    std::string err;
    std::vector<float> pixel_buf;
    if (!image_path.empty()) {
        siglip2::PreprocResult pp;
        float mean[3] = {0.5f, 0.5f, 0.5f};
        float std_v[3] = {0.5f, 0.5f, 0.5f};
        if (!siglip2::preprocess_image_file(
                image_path, max_num_patches, cfg.patch_size,
                1.0f / 255.0f, mean, std_v, pp, err)) {
            fprintf(stderr, "preprocess failed: %s\n", err.c_str());
            return 1;
        }
        pixel_buf = std::move(pp.pixel_values);
        n_h = pp.n_patches_h;
        n_w = pp.n_patches_w;
        fprintf(stderr, "preprocessed: grid=%dx%d (%d patches, max=%d)\n",
            n_h, n_w, n_h * n_w, max_num_patches);
    } else {
        if (n_h < 0 && n_w < 0) {
            const int side = (int)(std::round(std::sqrt((double)cfg.num_patches)));
            n_h = side;
            n_w = side;
        } else if (n_h < 0 || n_w < 0) {
            fprintf(stderr, "--shape must specify both H and W\n");
            return 2;
        }
        const int n_patches_inferred = n_h * n_w;
        if (n_patches < 0) n_patches = n_patches_inferred;
        if (n_patches != n_patches_inferred) {
            fprintf(stderr, "--n-patches=%d disagrees with --shape %d,%d (=%d).\n",
                n_patches, n_h, n_w, n_patches_inferred);
            return 2;
        }
        std::vector<uint8_t> raw;
        if (!read_file(pixel_path, raw, err)) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        const size_t expected = sizeof(float) * (size_t)n_patches * feat;
        if (raw.size() != expected) {
            fprintf(stderr,
                "pixel buffer size mismatch: got %zu bytes, expected %zu (n_patches=%d feat=%d)\n",
                raw.size(), expected, n_patches, feat);
            return 1;
        }
        pixel_buf.assign(reinterpret_cast<const float *>(raw.data()),
                         reinterpret_cast<const float *>(raw.data()) + raw.size() / sizeof(float));
    }

    std::vector<float> emb;
    if (!enc.encode(pixel_buf.data(), n_h, n_w, pooling, emb)) {
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
