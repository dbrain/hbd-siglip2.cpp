#include "siglip2_vision.h"

#include "gguf_loader.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace siglip2 {

namespace {

constexpr const char * KV_HIDDEN     = "siglip2.vision.embedding_length";
constexpr const char * KV_INTER      = "siglip2.vision.feed_forward_length";
constexpr const char * KV_HEADS      = "siglip2.vision.attention.head_count";
constexpr const char * KV_LAYERS     = "siglip2.vision.block_count";
constexpr const char * KV_PATCH      = "siglip2.vision.patch_size";
constexpr const char * KV_NUM_PATCH  = "siglip2.vision.num_patches";
constexpr const char * KV_NUM_CHAN   = "siglip2.vision.num_channels";
constexpr const char * KV_LN_EPS     = "siglip2.vision.layer_norm_eps";

std::string blk_name(int il, const char * suffix) {
    char buf[64];
    snprintf(buf, sizeof(buf), "v.blk.%d.%s", il, suffix);
    return buf;
}

struct Block {
    ggml_tensor * ln1_w  = nullptr; ggml_tensor * ln1_b  = nullptr;
    ggml_tensor * q_w    = nullptr; ggml_tensor * q_b    = nullptr;
    ggml_tensor * k_w    = nullptr; ggml_tensor * k_b    = nullptr;
    ggml_tensor * v_w    = nullptr; ggml_tensor * v_b    = nullptr;
    ggml_tensor * o_w    = nullptr; ggml_tensor * o_b    = nullptr;
    ggml_tensor * ln2_w  = nullptr; ggml_tensor * ln2_b  = nullptr;
    ggml_tensor * up_w   = nullptr; ggml_tensor * up_b   = nullptr;
    ggml_tensor * down_w = nullptr; ggml_tensor * down_b = nullptr;
};

// Siglip2MultiheadAttentionPoolingHead weights.
// HF uses nn.MultiheadAttention with packed in_proj_{weight,bias}; the converter
// splits these into separate q/k/v tensors so we can use the same attention
// pattern as encoder blocks.
struct ProbeHead {
    ggml_tensor * probe   = nullptr; // [1, 1, H]
    ggml_tensor * q_w     = nullptr; ggml_tensor * q_b     = nullptr;
    ggml_tensor * k_w     = nullptr; ggml_tensor * k_b     = nullptr;
    ggml_tensor * v_w     = nullptr; ggml_tensor * v_b     = nullptr;
    ggml_tensor * o_w     = nullptr; ggml_tensor * o_b     = nullptr;
    ggml_tensor * ln_w    = nullptr; ggml_tensor * ln_b    = nullptr;
    ggml_tensor * up_w    = nullptr; ggml_tensor * up_b    = nullptr;
    ggml_tensor * down_w  = nullptr; ggml_tensor * down_b  = nullptr;
};

ggml_tensor * dup_meta(
    ggml_context *      dst_ctx,
    ggml_context *      meta_ctx,
    const std::string & name) {
    ggml_tensor * meta = ggml_get_tensor(meta_ctx, name.c_str());
    if (!meta) return nullptr;
    ggml_tensor * out = ggml_dup_tensor(dst_ctx, meta);
    ggml_set_name(out, name.c_str());
    return out;
}

ggml_tensor * build_layernorm(
    ggml_context * ctx,
    ggml_tensor *  cur,
    ggml_tensor *  weight,
    ggml_tensor *  bias,
    float          eps) {
    cur = ggml_norm(ctx, cur, eps);
    if (weight) cur = ggml_mul(ctx, cur, weight);
    if (bias)   cur = ggml_add(ctx, cur, bias);
    return cur;
}

// One SigLIP2 encoder block: pre-LN attention + pre-LN MLP, both with residuals.
// Attention is fused via ggml_flash_attn_ext when use_fa=true so the per-layer
// (n_pos, n_pos, n_head) KQ matrix is never materialized — saves activation
// VRAM and turns the attn into one CUDA kernel launch instead of three.
ggml_tensor * build_block(
    ggml_context * ctx,
    const Block &  layer,
    ggml_tensor *  inp,
    int            n_pos,
    int            d_head,
    int            n_head,
    float          ln_eps,
    float          kq_scale,
    bool           use_fa) {
    // Self-attention
    ggml_tensor * residual = inp;
    ggml_tensor * cur = build_layernorm(ctx, inp, layer.ln1_w, layer.ln1_b, ln_eps);

    ggml_tensor * Q = ggml_add(ctx, ggml_mul_mat(ctx, layer.q_w, cur), layer.q_b);
    ggml_tensor * K = ggml_add(ctx, ggml_mul_mat(ctx, layer.k_w, cur), layer.k_b);
    ggml_tensor * V = ggml_add(ctx, ggml_mul_mat(ctx, layer.v_w, cur), layer.v_b);

    Q = ggml_reshape_3d(ctx, Q, d_head, n_head, n_pos);
    K = ggml_reshape_3d(ctx, K, d_head, n_head, n_pos);
    V = ggml_reshape_3d(ctx, V, d_head, n_head, n_pos);

    ggml_tensor * KQV;
    if (use_fa) {
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);                 // (d_head, n_pos, n_head)
        K = ggml_permute(ctx, K, 0, 2, 1, 3);                 // (d_head, n_pos, n_head)
        V = ggml_permute(ctx, V, 0, 2, 1, 3);                 // (d_head, n_pos, n_head) — NOT transposed
        K = ggml_cast(ctx, K, GGML_TYPE_F16);
        V = ggml_cast(ctx, V, GGML_TYPE_F16);
        KQV = ggml_flash_attn_ext(ctx, Q, K, V, /*mask*/ nullptr, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(KQV, GGML_PREC_F32);
        // FA result is already permuted: ne=[d_head, n_head, n_pos, 1]
        KQV = ggml_reshape_2d(ctx, KQV, d_head * n_head, n_pos);
    } else {
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);                 // (d_head, n_pos, n_head)
        K = ggml_permute(ctx, K, 0, 2, 1, 3);                 // (d_head, n_pos, n_head)
        V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3)); // (n_pos, d_head, n_head)
        ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
        KQ = ggml_soft_max_ext(ctx, KQ, /*mask*/ nullptr, kq_scale, 0.0f);
        KQV = ggml_mul_mat(ctx, V, KQ);                       // (d_head, n_pos_q, n_head)
        KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);             // (d_head, n_head, n_pos)
        KQV = ggml_cont_2d(ctx, KQV, d_head * n_head, n_pos); // (hidden, n_pos)
    }

    cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.o_w, KQV), layer.o_b);
    cur = ggml_add(ctx, cur, residual);
    residual = cur;

    // MLP
    cur = build_layernorm(ctx, cur, layer.ln2_w, layer.ln2_b, ln_eps);
    cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.up_w, cur), layer.up_b);
    cur = ggml_gelu(ctx, cur); // matches gelu_pytorch_tanh
    cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.down_w, cur), layer.down_b);

    cur = ggml_add(ctx, cur, residual);
    return cur;
}

// MultiheadAttentionPoolingHead. Cross-attention with a learnable probe (n_pos_q=1)
// against the last_hidden_state (n_pos_k=n_patches). Then layernorm + MLP residual.
// Returns the single pooled vector of shape (H, 1).
ggml_tensor * build_probe_head(
    ggml_context *      ctx,
    const ProbeHead &   head,
    ggml_tensor *       last_hidden,  // (H, n_pos)
    int                 n_pos,
    int                 H,
    int                 d_head,
    int                 n_head,
    float               ln_eps,
    float               kq_scale,
    bool                use_fa) {
    // Probe weight is stored (H, 1, 1) in GGUF (H is innermost). Build Q from it.
    // Treat probe as a single-token "input" of shape (H, 1).
    ggml_tensor * probe_in = ggml_reshape_2d(ctx, head.probe, H, 1);

    // Q = probe_in @ q_w.T + q_b  -> (H, 1)
    ggml_tensor * Q = ggml_add(ctx, ggml_mul_mat(ctx, head.q_w, probe_in), head.q_b);
    // K, V from last_hidden -> (H, n_pos)
    ggml_tensor * K = ggml_add(ctx, ggml_mul_mat(ctx, head.k_w, last_hidden), head.k_b);
    ggml_tensor * V = ggml_add(ctx, ggml_mul_mat(ctx, head.v_w, last_hidden), head.v_b);

    Q = ggml_reshape_3d(ctx, Q, d_head, n_head, 1);       // (d_head, n_head, 1)
    K = ggml_reshape_3d(ctx, K, d_head, n_head, n_pos);   // (d_head, n_head, n_pos)
    V = ggml_reshape_3d(ctx, V, d_head, n_head, n_pos);

    ggml_tensor * KQV;
    if (use_fa) {
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);             // (d_head, 1, n_head)
        K = ggml_permute(ctx, K, 0, 2, 1, 3);             // (d_head, n_pos, n_head)
        V = ggml_permute(ctx, V, 0, 2, 1, 3);             // (d_head, n_pos, n_head) — NOT transposed
        K = ggml_cast(ctx, K, GGML_TYPE_F16);
        V = ggml_cast(ctx, V, GGML_TYPE_F16);
        KQV = ggml_flash_attn_ext(ctx, Q, K, V, /*mask*/ nullptr, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(KQV, GGML_PREC_F32);
        // FA result: ne=[d_head, n_head, 1, 1]
        KQV = ggml_reshape_2d(ctx, KQV, d_head * n_head, 1);
    } else {
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);             // (d_head, 1, n_head)
        K = ggml_permute(ctx, K, 0, 2, 1, 3);             // (d_head, n_pos, n_head)
        V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));  // (n_pos, d_head, n_head)
        ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);                       // (n_pos, 1, n_head)
        KQ = ggml_soft_max_ext(ctx, KQ, /*mask*/ nullptr, kq_scale, 0.0f);
        KQV = ggml_mul_mat(ctx, V, KQ);                                   // (d_head, 1, n_head)
        KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);                         // (d_head, n_head, 1)
        KQV = ggml_cont_2d(ctx, KQV, d_head * n_head, 1);                 // (H, 1)
    }

    ggml_tensor * attn = ggml_add(ctx, ggml_mul_mat(ctx, head.o_w, KQV), head.o_b);  // (H, 1)

    // Per HF: residual = attn (no layernorm before MHA); LayerNorm wraps the post-MHA value;
    //         output = residual + MLP(LN(attn))
    ggml_tensor * normed = build_layernorm(ctx, attn, head.ln_w, head.ln_b, ln_eps);
    normed = ggml_add(ctx, ggml_mul_mat(ctx, head.up_w, normed), head.up_b);
    normed = ggml_gelu(ctx, normed);
    normed = ggml_add(ctx, ggml_mul_mat(ctx, head.down_w, normed), head.down_b);

    ggml_tensor * out = ggml_add(ctx, attn, normed);   // (H, 1)
    return out;
}

} // anonymous namespace

struct VisionEncoder::State {
    qwen3_tts::GGUFLoader loader;

    ggml_backend_t        backend     = nullptr;
    ggml_backend_t        backend_cpu = nullptr; // scheduler fallback when backend != CPU
    ggml_backend_buffer_t weights_buf = nullptr;
    ggml_context *        weights_ctx = nullptr;
    ggml_backend_sched_t  sched       = nullptr;

    // Weight tensors (in weights_ctx)
    ggml_tensor * patch_embd_w = nullptr;
    ggml_tensor * patch_embd_b = nullptr;
    ggml_tensor * pos_embd     = nullptr;
    ggml_tensor * post_ln_w    = nullptr;
    ggml_tensor * post_ln_b    = nullptr;

    std::vector<Block> blocks;
    ProbeHead          head;

    ~State() {
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        if (weights_buf) {
            ggml_backend_buffer_free(weights_buf);
        }
        if (weights_ctx) {
            ggml_free(weights_ctx);
        }
        if (backend_cpu && backend_cpu != backend) {
            ggml_backend_free(backend_cpu);
        }
        if (backend) {
            qwen3_tts::release_preferred_backend(backend);
        }
    }
};

VisionEncoder::VisionEncoder() = default;

VisionEncoder::~VisionEncoder() {
    close();
}

void VisionEncoder::close() {
    if (state_) {
        delete state_;
        state_ = nullptr;
    }
}

bool VisionEncoder::load(const std::string & gguf_path) {
    close();
    state_ = new State{};

    if (!state_->loader.open(gguf_path)) {
        error_msg_ = state_->loader.get_error();
        delete state_;
        state_ = nullptr;
        return false;
    }

    config_.hidden_size         = state_->loader.get_u32(KV_HIDDEN, 1152);
    config_.intermediate_size   = state_->loader.get_u32(KV_INTER,  4304);
    config_.num_attention_heads = state_->loader.get_u32(KV_HEADS,  16);
    config_.num_hidden_layers   = state_->loader.get_u32(KV_LAYERS, 27);
    config_.patch_size          = state_->loader.get_u32(KV_PATCH,  16);
    config_.num_patches         = state_->loader.get_u32(KV_NUM_PATCH, 256);
    config_.num_channels        = state_->loader.get_u32(KV_NUM_CHAN, 3);
    config_.layer_norm_eps      = state_->loader.get_f32(KV_LN_EPS, 1e-6f);

    if (config_.hidden_size <= 0 || config_.num_hidden_layers <= 0) {
        error_msg_ = "GGUF metadata missing required vision config keys";
        delete state_;
        state_ = nullptr;
        return false;
    }

    state_->backend = qwen3_tts::init_preferred_backend("siglip2-vision", &error_msg_);
    if (!state_->backend) {
        delete state_;
        state_ = nullptr;
        return false;
    }
    if (ggml_backend_dev_type(ggml_backend_get_device(state_->backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state_->backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    } else {
        state_->backend_cpu = state_->backend;
    }

    const int64_t n_total = state_->loader.get_n_tensors();
    struct ggml_init_params wp = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t)(n_total + 16),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    state_->weights_ctx = ggml_init(wp);
    if (!state_->weights_ctx) {
        error_msg_ = "ggml_init for weights_ctx failed";
        delete state_;
        state_ = nullptr;
        return false;
    }

    ggml_context * meta = state_->loader.get_meta_ctx();
    std::map<std::string, ggml_tensor *> tmap;

    auto add = [&](ggml_tensor * t, const char * name) -> bool {
        if (!t) {
            error_msg_ = std::string("Missing tensor in GGUF: ") + name;
            return false;
        }
        tmap[name] = t;
        return true;
    };

    state_->patch_embd_w = dup_meta(state_->weights_ctx, meta, "v.patch_embd.weight");
    state_->patch_embd_b = dup_meta(state_->weights_ctx, meta, "v.patch_embd.bias");
    state_->pos_embd     = dup_meta(state_->weights_ctx, meta, "v.position_embd.weight");
    state_->post_ln_w    = dup_meta(state_->weights_ctx, meta, "v.post_ln.weight");
    state_->post_ln_b    = dup_meta(state_->weights_ctx, meta, "v.post_ln.bias");
    if (!add(state_->patch_embd_w, "v.patch_embd.weight") ||
        !add(state_->patch_embd_b, "v.patch_embd.bias") ||
        !add(state_->pos_embd,     "v.position_embd.weight") ||
        !add(state_->post_ln_w,    "v.post_ln.weight") ||
        !add(state_->post_ln_b,    "v.post_ln.bias")) {
        delete state_;
        state_ = nullptr;
        return false;
    }

    // Probe pooling head tensors (optional — only present in full SigLIP2 models).
    {
        ProbeHead & h = state_->head;
        struct entry { ggml_tensor ** dst; const char * name; };
        entry entries[] = {
            {&h.probe,  "v.head.probe"},
            {&h.q_w,    "v.head.attn_q.weight"},   {&h.q_b,    "v.head.attn_q.bias"},
            {&h.k_w,    "v.head.attn_k.weight"},   {&h.k_b,    "v.head.attn_k.bias"},
            {&h.v_w,    "v.head.attn_v.weight"},   {&h.v_b,    "v.head.attn_v.bias"},
            {&h.o_w,    "v.head.attn_o.weight"},   {&h.o_b,    "v.head.attn_o.bias"},
            {&h.ln_w,   "v.head.ln.weight"},       {&h.ln_b,   "v.head.ln.bias"},
            {&h.up_w,   "v.head.ffn_up.weight"},   {&h.up_b,   "v.head.ffn_up.bias"},
            {&h.down_w, "v.head.ffn_down.weight"}, {&h.down_b, "v.head.ffn_down.bias"},
        };
        for (auto & e : entries) {
            *e.dst = dup_meta(state_->weights_ctx, meta, e.name);
            if (!add(*e.dst, e.name)) {
                delete state_;
                state_ = nullptr;
                return false;
            }
        }
    }

    state_->blocks.resize(config_.num_hidden_layers);
    for (int il = 0; il < config_.num_hidden_layers; ++il) {
        Block & b = state_->blocks[il];
        struct entry { ggml_tensor ** dst; const char * suffix; };
        entry entries[] = {
            {&b.ln1_w,  "ln1.weight"},   {&b.ln1_b,  "ln1.bias"},
            {&b.q_w,    "attn_q.weight"},{&b.q_b,    "attn_q.bias"},
            {&b.k_w,    "attn_k.weight"},{&b.k_b,    "attn_k.bias"},
            {&b.v_w,    "attn_v.weight"},{&b.v_b,    "attn_v.bias"},
            {&b.o_w,    "attn_o.weight"},{&b.o_b,    "attn_o.bias"},
            {&b.ln2_w,  "ln2.weight"},   {&b.ln2_b,  "ln2.bias"},
            {&b.up_w,   "ffn_up.weight"},{&b.up_b,   "ffn_up.bias"},
            {&b.down_w, "ffn_down.weight"},{&b.down_b,"ffn_down.bias"},
        };
        for (auto & e : entries) {
            std::string nm = blk_name(il, e.suffix);
            *e.dst = dup_meta(state_->weights_ctx, meta, nm);
            if (!add(*e.dst, nm.c_str())) {
                delete state_;
                state_ = nullptr;
                return false;
            }
        }
    }

    // Allocate backend buffer + load weight data.
    if (!qwen3_tts::load_tensor_data_from_file(
            gguf_path,
            state_->loader.get_ctx(),
            state_->weights_ctx,
            tmap,
            state_->weights_buf,
            error_msg_,
            ggml_backend_dev_type(ggml_backend_get_device(state_->backend)))) {
        delete state_;
        state_ = nullptr;
        return false;
    }

    // Compute scheduler.
    {
        std::vector<ggml_backend_t> backends;
        backends.push_back(state_->backend);
        if (state_->backend_cpu && state_->backend_cpu != state_->backend) {
            backends.push_back(state_->backend_cpu);
        }
        std::vector<ggml_backend_buffer_type_t> bufts;
        for (auto b : backends) bufts.push_back(ggml_backend_get_default_buffer_type(b));
        const int max_nodes = 2048;
        state_->sched = ggml_backend_sched_new(
            backends.data(), bufts.data(), (int)backends.size(),
            max_nodes, /*parallel=*/false, /*op_offload=*/true);
        if (!state_->sched) {
            error_msg_ = "ggml_backend_sched_new failed";
            delete state_;
            state_ = nullptr;
            return false;
        }
    }

    return true;
}

bool VisionEncoder::encode(
    const float *        pixel_values,
    int                  n_patches_h,
    int                  n_patches_w,
    Pooling              pooling,
    std::vector<float> & out_embedding) {
    if (!state_) {
        error_msg_ = "VisionEncoder not loaded";
        return false;
    }
    if (n_patches_h <= 0 || n_patches_w <= 0) {
        error_msg_ = "n_patches_h and n_patches_w must be positive";
        return false;
    }
    const int n_per_side = (int)std::round(std::sqrt((double)config_.num_patches));
    if (n_per_side * n_per_side != config_.num_patches) {
        error_msg_ = "config.num_patches must be a perfect square (native grid)";
        return false;
    }

    const int   H        = config_.hidden_size;
    const int   n_head   = config_.num_attention_heads;
    const int   d_head   = H / n_head;
    const int   n_pos    = n_patches_h * n_patches_w;
    const int   feat     = config_.num_channels * config_.patch_size * config_.patch_size;
    const float kq_scale = 1.0f / std::sqrt((float)d_head);

    const int    max_nodes = 2048;
    const size_t arena_sz  =
        ggml_tensor_overhead() * 4096 +
        ggml_graph_overhead_custom(max_nodes, false);
    std::vector<uint8_t> arena(arena_sz);
    struct ggml_init_params gp = {
        /*.mem_size   =*/ arena.size(),
        /*.mem_buffer =*/ arena.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) {
        error_msg_ = "ggml_init for graph_ctx failed";
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);

    // Input: pixel_values [feat, n_pos] in ggml convention (ne[0]=feat innermost).
    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feat, n_pos);
    ggml_set_name(inp, "pixel_values");
    ggml_set_input(inp);

    // Patch embed: Linear(feat -> H). Weight stored (feat, H); mul_mat -> (H, n_pos).
    ggml_tensor * x = ggml_mul_mat(ctx, state_->patch_embd_w, inp);
    x = ggml_add(ctx, x, state_->patch_embd_b);

    // Position embedding: bilinear interpolate from native (n_per_side, n_per_side, H)
    // to target (n_patches_w, n_patches_h, H), then add. Skip the interpolation when
    // the target matches native (avoids small numerical noise from the resampler).
    ggml_tensor * pos_embd = state_->pos_embd; // (H, num_patches)
    if (n_patches_h != n_per_side || n_patches_w != n_per_side) {
        pos_embd = ggml_reshape_3d(ctx, pos_embd, H, n_per_side, n_per_side);
        pos_embd = ggml_permute(ctx, pos_embd, 2, 0, 1, 3);              // (n_per_side, n_per_side, H)
        // BILINEAR + ANTIALIAS matches HF Siglip2VisionEmbeddings.resize_positional_embeddings
        // exactly (which uses F.interpolate(mode='bilinear', antialias=True)).
        pos_embd = ggml_interpolate(
            ctx, pos_embd,
            n_patches_w, n_patches_h, H, 1,
            GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS);       // (n_w, n_h, H)
        pos_embd = ggml_permute(ctx, pos_embd, 1, 2, 0, 3);              // (H, n_w, n_h)
        pos_embd = ggml_cont_2d(ctx, pos_embd, H, n_pos);                // (H, n_pos)
    }
    x = ggml_add(ctx, x, pos_embd);

    // Flash attention is enabled by default — saves the per-layer KQ activation
    // (n_pos² × n_head × 4 bytes). For SigLIP2's d_head=72 the CUDA dispatcher
    // routes to the tile kernel (MMA paths skip 72), but it's still a clear win
    // on activation memory and end-to-end perf vs vanilla 3-op attention.
    // SIGLIP2_DISABLE_FA=1 falls back if a parity issue surfaces.
    static const bool use_fa = std::getenv("SIGLIP2_DISABLE_FA") == nullptr;

    for (int il = 0; il < config_.num_hidden_layers; ++il) {
        x = build_block(ctx, state_->blocks[il], x, n_pos, d_head, n_head, config_.layer_norm_eps, kq_scale, use_fa);
    }

    x = build_layernorm(ctx, x, state_->post_ln_w, state_->post_ln_b, config_.layer_norm_eps);

    ggml_tensor * pooled = nullptr;
    if (pooling == Pooling::MEAN) {
        // Mean-pool over patches: x is (H, n_pos); transpose to (n_pos, H) so ggml_mean reduces n_pos.
        pooled = ggml_cont(ctx, ggml_transpose(ctx, x));
        pooled = ggml_mean(ctx, pooled);
    } else { // PROBE
        pooled = build_probe_head(
            ctx, state_->head, x,
            n_pos, H, d_head, n_head, config_.layer_norm_eps, kq_scale, use_fa);
    }
    ggml_set_name(pooled, "pooled");
    ggml_set_output(pooled);

    ggml_build_forward_expand(gf, pooled);

    if (!ggml_backend_sched_alloc_graph(state_->sched, gf)) {
        ggml_free(ctx);
        error_msg_ = "ggml_backend_sched_alloc_graph failed";
        return false;
    }

    ggml_backend_tensor_set(inp, pixel_values, 0, sizeof(float) * (size_t)feat * n_pos);

    enum ggml_status st = ggml_backend_sched_graph_compute(state_->sched, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(state_->sched);
        ggml_free(ctx);
        error_msg_ = std::string("graph compute failed status=") + std::to_string((int)st);
        return false;
    }

    out_embedding.assign((size_t)H, 0.0f);
    ggml_backend_tensor_get(pooled, out_embedding.data(), 0, sizeof(float) * (size_t)H);

    ggml_backend_sched_reset(state_->sched);
    ggml_free(ctx);
    return true;
}

} // namespace siglip2
