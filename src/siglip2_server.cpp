// SigLIP2 HTTP server — drop-in replacement for kobbler-vision's FastAPI service.
//
// Endpoints (all match docker/kobbler-vision/server.py request/response shape):
//   GET  /health
//   POST /unload                       (no-op; lifecycle here is in-process)
//   POST /v1/embeddings                (multipart images)
//   POST /v1/classify                  (multipart images + form prompts)
//   POST /v1/classify_from_embeddings  (JSON {image_embeddings, prompts})
//   POST /v1/text_embeddings           (form prompts)

#include "siglip2_vision.h"
#include "siglip2_text.h"
#include "siglip2_tokenizer.h"
#include "siglip2_score.h"
#include "siglip2_preproc.h"

// stb_image is already linked into siglip2_preproc as the implementation;
// here we only need extern declarations to call stbi_load_from_memory.
#include "stb_image.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

// ---- config -----------------------------------------------------------------

struct ServerConfig {
    std::string model_path;
    std::string tokenizer_path;
    std::string host = "0.0.0.0";
    int         port = 8890;
    int         default_max_num_patches = 729;
    int         idle_unload_seconds = 0;     // 0 = never
    bool        lazy_load = false;
};

ServerConfig parse_args(int argc, char ** argv) {
    ServerConfig c;
    auto env = [](const char * key, const char * def) {
        const char * v = std::getenv(key);
        return v ? std::string(v) : std::string(def);
    };
    c.model_path     = env("MODEL_PATH", "");
    c.tokenizer_path = env("TOKENIZER_PATH", "");
    c.host           = env("HOST", "0.0.0.0");
    c.port           = std::atoi(env("PORT", "8890").c_str());
    c.default_max_num_patches = std::atoi(env("DEFAULT_MAX_NUM_PATCHES", "729").c_str());
    c.idle_unload_seconds     = std::atoi(env("IDLE_UNLOAD_SECONDS", "0").c_str());
    c.lazy_load               = !env("LAZY_LOAD", "").empty() &&
                                env("LAZY_LOAD", "") != "0" && env("LAZY_LOAD", "") != "false";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto take = [&](const char * flag, std::string & dst) {
            if (a == flag && i + 1 < argc) { dst = argv[++i]; return true; }
            return false;
        };
        if      (take("--model", c.model_path)) {}
        else if (take("--tokenizer", c.tokenizer_path)) {}
        else if (take("--host", c.host)) {}
        else if (a == "--port" && i + 1 < argc) c.port = std::atoi(argv[++i]);
        else if (a == "--default-max-num-patches" && i + 1 < argc) c.default_max_num_patches = std::atoi(argv[++i]);
        else if (a == "--idle-unload-seconds"     && i + 1 < argc) c.idle_unload_seconds     = std::atoi(argv[++i]);
        else if (a == "--lazy-load") c.lazy_load = true;
        else if (a == "--help" || a == "-h") {
            fprintf(stderr,
                "siglip2-server — kobbler-vision-compatible HTTP service\n\n"
                "Usage: %s --model <gguf> --tokenizer <spm.model> [--port 8890] [--lazy-load]\n"
                "Env: MODEL_PATH TOKENIZER_PATH HOST PORT DEFAULT_MAX_NUM_PATCHES IDLE_UNLOAD_SECONDS LAZY_LOAD\n",
                argv[0]);
            std::exit(0);
        }
    }
    return c;
}

// ---- L2 normalize -----------------------------------------------------------

void l2_normalize(std::vector<float> & v) {
    double s = 0.0;
    for (float x : v) s += (double)x * x;
    const float inv = (float)(1.0 / (std::sqrt(s) + 1e-12));
    for (auto & x : v) x *= inv;
}

// ---- ServerState ------------------------------------------------------------

struct ServerState {
    ServerConfig                cfg;
    std::mutex                  encode_mutex;
    std::unique_ptr<siglip2::VisionEncoder> vision;
    std::unique_ptr<siglip2::TextEncoder>   text;
    std::unique_ptr<siglip2::Tokenizer>     tokenizer;
    siglip2::ScoreParams        score_params;
    std::atomic<bool>           loaded{false};
    std::atomic<long long>      last_request_ms{0};

    bool load_model(std::string & err) {
        std::lock_guard<std::mutex> lock(encode_mutex);
        if (loaded.load()) return true;
        if (cfg.model_path.empty() || cfg.tokenizer_path.empty()) {
            err = "model and tokenizer paths required";
            return false;
        }
        vision    = std::make_unique<siglip2::VisionEncoder>();
        text      = std::make_unique<siglip2::TextEncoder>();
        tokenizer = std::make_unique<siglip2::Tokenizer>();
        if (!vision->load(cfg.model_path)) {
            err = "vision load: " + vision->last_error();
            return false;
        }
        if (!text->load(cfg.model_path)) {
            err = "text load: " + text->last_error();
            return false;
        }
        if (!tokenizer->load(cfg.tokenizer_path)) {
            err = "tokenizer load: " + tokenizer->last_error();
            return false;
        }
        if (!siglip2::read_score_params(cfg.model_path, score_params, err)) {
            return false;
        }
        loaded.store(true);
        fprintf(stderr, "[siglip2-server] model loaded: %s\n", cfg.model_path.c_str());
        return true;
    }

    void unload() {
        std::lock_guard<std::mutex> lock(encode_mutex);
        if (!loaded.load()) return;
        vision.reset();
        text.reset();
        tokenizer.reset();
        loaded.store(false);
        fprintf(stderr, "[siglip2-server] model unloaded\n");
    }

    long long now_ms() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void touch() { last_request_ms.store(now_ms()); }
};

// ---- helpers ----------------------------------------------------------------

// Decode an image (jpeg/png/...) from a memory buffer. Resulting RGB is uint8 row-major HWC.
bool decode_image(const std::string & bytes, std::vector<uint8_t> & rgb, int & h, int & w, std::string & err) {
    int channels = 0;
    uint8_t * px = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc *>(bytes.data()), (int)bytes.size(),
        &w, &h, &channels, /*req_comp=*/3);
    if (!px) {
        err = std::string("stbi_load_from_memory failed: ") + (stbi_failure_reason() ? stbi_failure_reason() : "?");
        return false;
    }
    rgb.assign(px, px + (size_t)h * w * 3);
    stbi_image_free(px);
    return true;
}

bool encode_image(
    ServerState &       st,
    const std::string & bytes,
    int                 max_num_patches,
    siglip2::Pooling    pooling,
    std::vector<float> & out_emb,
    std::string &       err) {
    std::vector<uint8_t> rgb;
    int h = 0, w = 0;
    if (!decode_image(bytes, rgb, h, w, err)) return false;

    siglip2::PreprocResult pp;
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float std_v[3] = {0.5f, 0.5f, 0.5f};
    if (!siglip2::preprocess_image_rgb(rgb.data(), h, w, max_num_patches,
                                       st.vision->config().patch_size,
                                       1.0f / 255.0f, mean, std_v, pp, err)) {
        return false;
    }
    if (!st.vision->encode(pp.pixel_values.data(), pp.n_patches_h, pp.n_patches_w, pooling, out_emb)) {
        err = "vision encode: " + st.vision->last_error();
        return false;
    }
    return true;
}

bool encode_text(
    ServerState &       st,
    const std::string & prompt,
    std::vector<float> & out_emb,
    std::string &       err) {
    std::vector<int32_t> ids, mask;
    const int seq = st.text->config().max_position_embeddings;
    if (!st.tokenizer->encode(prompt, seq, ids, mask)) {
        err = "tokenize: " + st.tokenizer->last_error();
        return false;
    }
    if (!st.text->encode(ids.data(), mask.data(), out_emb)) {
        err = "text encode: " + st.text->last_error();
        return false;
    }
    return true;
}

// Collect form values for `name` from both multipart fields (req.files) and
// form-urlencoded body / URL params (req.params). FastAPI's list[str] = Form(...)
// can arrive either way depending on Content-Type the client sends.
std::vector<std::string> collect_field(const httplib::Request & req, const std::string & name) {
    std::vector<std::string> out;
    auto fr = req.files.equal_range(name);
    for (auto it = fr.first; it != fr.second; ++it) {
        out.push_back(it->second.content);
    }
    auto pr = req.params.equal_range(name);
    for (auto it = pr.first; it != pr.second; ++it) {
        out.push_back(it->second);
    }
    return out;
}

siglip2::Pooling parse_pooling(const std::string & s, std::string & err) {
    if (s.empty() || s == "pooler" || s == "probe") return siglip2::Pooling::PROBE;
    if (s == "mean") return siglip2::Pooling::MEAN;
    err = "pooling must be one of: pooler, probe, mean (got '" + s + "')";
    return siglip2::Pooling::PROBE;
}

void send_json(httplib::Response & res, int status, const json & body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

} // namespace

int main(int argc, char ** argv) {
    ServerConfig cfg = parse_args(argc, argv);
    if (cfg.model_path.empty() || cfg.tokenizer_path.empty()) {
        fprintf(stderr, "Both --model and --tokenizer (or MODEL_PATH/TOKENIZER_PATH env) are required.\n");
        return 2;
    }
    ServerState st;
    st.cfg = cfg;
    if (!cfg.lazy_load) {
        std::string err;
        if (!st.load_model(err)) {
            fprintf(stderr, "load failed: %s\n", err.c_str());
            return 1;
        }
    }

    httplib::Server srv;
    srv.set_keep_alive_max_count(20);
    srv.set_read_timeout(60);
    srv.set_write_timeout(60);

    // -- /health
    srv.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        json body = {
            {"status", "ok"},
            {"model", cfg.model_path},
            {"model_loaded", st.loaded.load()},
        };
        send_json(res, 200, body);
    });

    // -- /unload
    srv.Post("/unload", [&](const httplib::Request &, httplib::Response & res) {
        st.unload();
        send_json(res, 200, json{{"status", "unloaded"}});
    });

    // ensure-model helper
    auto ensure = [&](httplib::Response & res) -> bool {
        st.touch();
        if (!st.loaded.load()) {
            std::string err;
            if (!st.load_model(err)) {
                send_json(res, 503, json{{"error", err}});
                return false;
            }
        }
        return true;
    };

    // -- /v1/embeddings
    srv.Post("/v1/embeddings", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ensure(res)) return;
        auto images = req.get_file_values("images");
        if (images.empty()) {
            send_json(res, 400, json{{"error", "No images provided"}});
            return;
        }
        std::string err;
        const std::string pooling_str = req.get_param_value("pooling");
        siglip2::Pooling pooling = parse_pooling(pooling_str, err);
        if (!err.empty()) { send_json(res, 400, json{{"error", err}}); return; }
        int max_num_patches = cfg.default_max_num_patches;
        if (req.has_param("max_num_patches")) {
            const std::string v = req.get_param_value("max_num_patches");
            if (!v.empty()) max_num_patches = std::atoi(v.c_str());
        }
        const bool return_last_hidden = req.get_param_value("return_last_hidden") == "true";
        if (return_last_hidden) {
            // Not implemented in M4 — would require a separate encode-with-hidden path.
            send_json(res, 501, json{{"error", "return_last_hidden not implemented"}});
            return;
        }

        std::vector<std::vector<float>> embs;
        embs.reserve(images.size());
        {
            std::lock_guard<std::mutex> lock(st.encode_mutex);
            for (const auto & img : images) {
                std::vector<float> emb;
                if (!encode_image(st, img.content, max_num_patches, pooling, emb, err)) {
                    send_json(res, 500, json{{"error", err}});
                    return;
                }
                l2_normalize(emb);
                embs.emplace_back(std::move(emb));
            }
        }
        send_json(res, 200, json{{"embeddings", embs}});
    });

    // -- /v1/text_embeddings
    srv.Post("/v1/text_embeddings", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ensure(res)) return;
        // Accept multiple `prompts` form fields per kobbler-vision (FastAPI list[str] = Form(...)).
        std::vector<std::string> prompts = collect_field(req, "prompts");
        if (prompts.empty()) { send_json(res, 400, json{{"error", "No prompts provided"}}); return; }

        std::vector<std::vector<float>> embs;
        embs.reserve(prompts.size());
        std::string err;
        {
            std::lock_guard<std::mutex> lock(st.encode_mutex);
            for (const auto & p : prompts) {
                std::vector<float> emb;
                if (!encode_text(st, p, emb, err)) {
                    send_json(res, 500, json{{"error", err}});
                    return;
                }
                l2_normalize(emb);
                embs.emplace_back(std::move(emb));
            }
        }
        send_json(res, 200, json{{"embeddings", embs}});
    });

    // -- /v1/classify
    srv.Post("/v1/classify", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ensure(res)) return;
        auto images = req.get_file_values("images");
        std::vector<std::string> prompts = collect_field(req, "prompts");
        if (images.empty())  { send_json(res, 400, json{{"error", "No images provided"}}); return; }
        if (prompts.empty()) { send_json(res, 400, json{{"error", "No prompts provided"}}); return; }

        const bool return_logits = req.get_param_value("return_logits") == "true";
        int max_num_patches = cfg.default_max_num_patches;
        if (req.has_param("max_num_patches")) {
            const std::string v = req.get_param_value("max_num_patches");
            if (!v.empty()) max_num_patches = std::atoi(v.c_str());
        }

        const int H = (int)st.vision->config().hidden_size;
        const int n_img = (int)images.size();
        const int n_txt = (int)prompts.size();
        std::vector<float> img_buf((size_t)n_img * H);
        std::vector<float> txt_buf((size_t)n_txt * H);

        std::string err;
        {
            std::lock_guard<std::mutex> lock(st.encode_mutex);
            for (int i = 0; i < n_img; ++i) {
                std::vector<float> emb;
                if (!encode_image(st, images[i].content, max_num_patches, siglip2::Pooling::PROBE, emb, err)) {
                    send_json(res, 500, json{{"error", err}}); return;
                }
                std::memcpy(img_buf.data() + (size_t)i * H, emb.data(), sizeof(float) * H);
            }
            for (int j = 0; j < n_txt; ++j) {
                std::vector<float> emb;
                if (!encode_text(st, prompts[j], emb, err)) {
                    send_json(res, 500, json{{"error", err}}); return;
                }
                std::memcpy(txt_buf.data() + (size_t)j * H, emb.data(), sizeof(float) * H);
            }
        }
        std::vector<float> logits((size_t)n_img * n_txt);
        std::vector<float> probs((size_t)n_img * n_txt);
        siglip2::score_image_text(
            img_buf.data(), n_img, txt_buf.data(), n_txt, H,
            st.score_params, logits.data(), probs.data());

        // scores: (n_img, n_txt)
        std::vector<std::vector<float>> scores_out(n_img, std::vector<float>(n_txt));
        std::vector<std::vector<float>> logits_out(n_img, std::vector<float>(n_txt));
        for (int i = 0; i < n_img; ++i)
            for (int j = 0; j < n_txt; ++j) {
                scores_out[i][j] = probs[(size_t)i * n_txt + j];
                logits_out[i][j] = logits[(size_t)i * n_txt + j];
            }
        json body = {{"scores", scores_out}};
        if (return_logits) body["logits"] = logits_out;
        send_json(res, 200, body);
    });

    // -- /v1/classify_from_embeddings (JSON body)
    srv.Post("/v1/classify_from_embeddings", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ensure(res)) return;
        json body;
        try {
            body = json::parse(req.body);
        } catch (std::exception & e) {
            send_json(res, 400, json{{"error", std::string("invalid JSON: ") + e.what()}});
            return;
        }
        if (!body.contains("image_embeddings") || !body.contains("prompts")) {
            send_json(res, 400, json{{"error", "JSON must contain image_embeddings and prompts"}});
            return;
        }
        const bool return_logits = body.value("return_logits", false);

        const int H = (int)st.vision->config().hidden_size;
        const auto & imgs_json = body["image_embeddings"];
        const auto & prompts_json = body["prompts"];
        const int n_img = (int)imgs_json.size();
        const int n_txt = (int)prompts_json.size();

        std::vector<float> img_buf((size_t)n_img * H);
        for (int i = 0; i < n_img; ++i) {
            const auto & row = imgs_json[i];
            if ((int)row.size() != H) {
                send_json(res, 400, json{{"error", "image_embedding row has wrong dim"}});
                return;
            }
            for (int c = 0; c < H; ++c) img_buf[(size_t)i * H + c] = row[c].get<float>();
        }

        std::vector<float> txt_buf((size_t)n_txt * H);
        std::string err;
        {
            std::lock_guard<std::mutex> lock(st.encode_mutex);
            for (int j = 0; j < n_txt; ++j) {
                std::vector<float> emb;
                if (!encode_text(st, prompts_json[j].get<std::string>(), emb, err)) {
                    send_json(res, 500, json{{"error", err}}); return;
                }
                std::memcpy(txt_buf.data() + (size_t)j * H, emb.data(), sizeof(float) * H);
            }
        }

        std::vector<float> logits((size_t)n_img * n_txt);
        std::vector<float> probs((size_t)n_img * n_txt);
        siglip2::score_image_text(
            img_buf.data(), n_img, txt_buf.data(), n_txt, H,
            st.score_params, logits.data(), probs.data());

        std::vector<std::vector<float>> scores_out(n_img, std::vector<float>(n_txt));
        std::vector<std::vector<float>> logits_out(n_img, std::vector<float>(n_txt));
        for (int i = 0; i < n_img; ++i)
            for (int j = 0; j < n_txt; ++j) {
                scores_out[i][j] = probs[(size_t)i * n_txt + j];
                logits_out[i][j] = logits[(size_t)i * n_txt + j];
            }
        json out = {{"scores", scores_out}};
        if (return_logits) out["logits"] = logits_out;
        send_json(res, 200, out);
    });

    fprintf(stderr, "[siglip2-server] listening on %s:%d (model=%s)\n",
        cfg.host.c_str(), cfg.port, cfg.model_path.c_str());
    if (!srv.listen(cfg.host.c_str(), cfg.port)) {
        fprintf(stderr, "listen failed\n");
        return 1;
    }
    return 0;
}
