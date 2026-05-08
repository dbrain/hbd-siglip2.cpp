#include "siglip2_text.h"

#include "gguf_loader.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace siglip2 {

namespace {

constexpr const char * KV_HIDDEN     = "siglip2.text.embedding_length";
constexpr const char * KV_INTER      = "siglip2.text.feed_forward_length";
constexpr const char * KV_HEADS      = "siglip2.text.attention.head_count";
constexpr const char * KV_LAYERS     = "siglip2.text.block_count";
constexpr const char * KV_VOCAB      = "siglip2.text.vocab_size";
constexpr const char * KV_MAX_POS    = "siglip2.text.max_position_embeddings";
constexpr const char * KV_PROJ       = "siglip2.text.projection_size";
constexpr const char * KV_LN_EPS     = "siglip2.text.layer_norm_eps";

std::string blk_name(int il, const char * suffix) {
    char buf[64];
    snprintf(buf, sizeof(buf), "t.blk.%d.%s", il, suffix);
    return buf;
}

struct Block {
    ggml_tensor * ln1_w  = nullptr; ggml_tensor * ln1_b  = nullptr;
    // Fused Q/K/V: weight is (H, 3*H), bias is (3*H,). Replaces three separate
    // q/k/v projections with one wider mul_mat that we slice into Q, K, V views
    // before the attention. Saves 2 launches per layer.
    ggml_tensor * qkv_w  = nullptr; ggml_tensor * qkv_b  = nullptr;
    ggml_tensor * o_w    = nullptr; ggml_tensor * o_b    = nullptr;
    ggml_tensor * ln2_w  = nullptr; ggml_tensor * ln2_b  = nullptr;
    ggml_tensor * up_w   = nullptr; ggml_tensor * up_b   = nullptr;
    ggml_tensor * down_w = nullptr; ggml_tensor * down_b = nullptr;
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
    ggml_context * ctx, ggml_tensor * cur,
    ggml_tensor * weight, ggml_tensor * bias, float eps) {
    cur = ggml_norm(ctx, cur, eps);
    if (weight) cur = ggml_mul(ctx, cur, weight);
    if (bias)   cur = ggml_add(ctx, cur, bias);
    return cur;
}

// SigLIP2 encoder block. Same as the vision tower's, but takes an explicit
// attention mask for padding-aware self-attention. Uses ggml_flash_attn_ext
// when use_fa=true. fa_mask_f16 must be the F16 cast of attn_mask (or nullptr
// if attn_mask is nullptr) — passed in instead of being built here so that
// the same cast is reused across all layers.
ggml_tensor * build_block(
    ggml_context * ctx,
    const Block &  layer,
    ggml_tensor *  inp,
    ggml_tensor *  attn_mask,    // (n_pos_k, n_pos_q) F32 with 0/-inf, may be nullptr
    ggml_tensor *  fa_mask_f16,  // same as attn_mask but F16-cast for FA path
    int            n_pos,
    int            d_head,
    int            n_head,
    float          ln_eps,
    float          kq_scale,
    bool           use_fa) {
    ggml_tensor * residual = inp;
    ggml_tensor * cur = build_layernorm(ctx, inp, layer.ln1_w, layer.ln1_b, ln_eps);

    // Fused Q/K/V: one mul_mat over weight (H, 3*H), one bias add over (3*H,).
    // Result has ne=[3*H, n_pos]. Q/K/V are 3D *strided* views into it (no
    // ggml_reshape — the views aren't contiguous along positions because Q/K/V
    // are interleaved in the OUT axis). The downstream ggml_cont before
    // ggml_pad materializes them contiguously, same as the pre-fusion path.
    const int    H  = d_head * n_head;
    const size_t es = sizeof(float);  // qkv is F32 (mul_mat output)
    ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, layer.qkv_w, cur), layer.qkv_b);
    ggml_tensor * Q = ggml_view_3d(ctx, qkv, d_head, n_head, n_pos,
        /*nb1=*/d_head * es, /*nb2=*/3 * H * es, /*offset=*/0 * H * es);
    ggml_tensor * K = ggml_view_3d(ctx, qkv, d_head, n_head, n_pos,
        /*nb1=*/d_head * es, /*nb2=*/3 * H * es, /*offset=*/1 * H * es);
    ggml_tensor * V = ggml_view_3d(ctx, qkv, d_head, n_head, n_pos,
        /*nb1=*/d_head * es, /*nb2=*/3 * H * es, /*offset=*/2 * H * es);

    ggml_tensor * KQV;
    if (use_fa) {
        // d_head=72 needs zero-padding to 80 to be tensor-core MMA eligible.
        // ggml's CUDA pad op only takes F32, so we pad before the F16 cast.
        constexpr int FA_TC_ALIGN = 16;
        const int d_pad = (d_head + FA_TC_ALIGN - 1) & ~(FA_TC_ALIGN - 1);
        const int pad   = d_pad - d_head;

        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);          // NOT transposed under FA
        if (pad > 0) {
            Q = ggml_pad(ctx, ggml_cont(ctx, Q), pad, 0, 0, 0);
            K = ggml_pad(ctx, ggml_cont(ctx, K), pad, 0, 0, 0);
            V = ggml_pad(ctx, ggml_cont(ctx, V), pad, 0, 0, 0);
        }
        ggml_tensor * K_f16 = ggml_cast(ctx, K, GGML_TYPE_F16);
        ggml_tensor * V_f16 = ggml_cast(ctx, V, GGML_TYPE_F16);
        KQV = ggml_flash_attn_ext(ctx, Q, K_f16, V_f16, fa_mask_f16, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(KQV, GGML_PREC_F32);
        if (pad > 0) {
            // Slice ne[0]=d_pad back to real d_head and rebuild contiguous.
            KQV = ggml_view_3d(ctx, KQV, d_head, n_head, n_pos, KQV->nb[1], KQV->nb[2], 0);
            KQV = ggml_cont(ctx, KQV);
        }
        KQV = ggml_reshape_2d(ctx, KQV, d_head * n_head, n_pos);
    } else {
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));
        ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
        KQ = ggml_soft_max_ext(ctx, KQ, attn_mask, kq_scale, 0.0f);
        KQV = ggml_mul_mat(ctx, V, KQ);
        KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
        KQV = ggml_cont_2d(ctx, KQV, d_head * n_head, n_pos);
    }

    cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.o_w, KQV), layer.o_b);
    cur = ggml_add(ctx, cur, residual);
    residual = cur;

    cur = build_layernorm(ctx, cur, layer.ln2_w, layer.ln2_b, ln_eps);
    cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.up_w, cur), layer.up_b);
    cur = ggml_gelu(ctx, cur);
    cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.down_w, cur), layer.down_b);

    cur = ggml_add(ctx, cur, residual);
    return cur;
}

// Batched variant of build_block. All activations are 3D (H, n_pos, n_batch);
// QKV views and FA inputs are 4D with ne[3]=n_batch. For n_batch=1 the memory
// layout is byte-identical to the unbatched path (the trailing 1-dim doesn't
// add bytes), but the explicit 4D-ness lets ggml's CUDA backend dispatch the
// batched ops correctly when n_batch>1.
//
// Megakernel hooks (A0/A1/A2/A3) gate on ne[2]==1 && ne[3]==1 — so they
// cleanly fall back to ggml's stock dispatch when n_batch>1, but still fire
// when n_batch==1. The parity guarantee is the same as the unbatched path.
ggml_tensor * build_block_batched(
    ggml_context * ctx,
    const Block &  layer,
    ggml_tensor *  inp,            // (H, n_pos, n_batch)
    ggml_tensor *  fa_mask_f16,    // (n_pos, n_pos, 1, n_batch) F16, may be nullptr
    int            n_pos,
    int            n_batch,
    int            d_head,
    int            n_head,
    float          ln_eps,
    float          kq_scale,
    bool           use_fa) {
    const int    H  = d_head * n_head;
    const size_t es = sizeof(float);

    ggml_tensor * residual = inp;
    ggml_tensor * cur = build_layernorm(ctx, inp, layer.ln1_w, layer.ln1_b, ln_eps);

    ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, layer.qkv_w, cur), layer.qkv_b);
    // qkv: (3*H, n_pos, n_batch). Q/K/V are 4D strided views into it.
    ggml_tensor * Q = ggml_view_4d(ctx, qkv, d_head, n_head, n_pos, n_batch,
        d_head * es, 3 * H * es, 3 * H * (size_t)n_pos * es, 0 * H * es);
    ggml_tensor * K = ggml_view_4d(ctx, qkv, d_head, n_head, n_pos, n_batch,
        d_head * es, 3 * H * es, 3 * H * (size_t)n_pos * es, 1 * H * es);
    ggml_tensor * V = ggml_view_4d(ctx, qkv, d_head, n_head, n_pos, n_batch,
        d_head * es, 3 * H * es, 3 * H * (size_t)n_pos * es, 2 * H * es);

    ggml_tensor * KQV;
    if (use_fa) {
        constexpr int FA_TC_ALIGN = 16;
        const int d_pad = (d_head + FA_TC_ALIGN - 1) & ~(FA_TC_ALIGN - 1);
        const int pad   = d_pad - d_head;

        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);
        if (pad > 0) {
            Q = ggml_pad(ctx, ggml_cont(ctx, Q), pad, 0, 0, 0);
            K = ggml_pad(ctx, ggml_cont(ctx, K), pad, 0, 0, 0);
            V = ggml_pad(ctx, ggml_cont(ctx, V), pad, 0, 0, 0);
        }
        ggml_tensor * K_f16 = ggml_cast(ctx, K, GGML_TYPE_F16);
        ggml_tensor * V_f16 = ggml_cast(ctx, V, GGML_TYPE_F16);
        KQV = ggml_flash_attn_ext(ctx, Q, K_f16, V_f16, fa_mask_f16, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(KQV, GGML_PREC_F32);
        // FA result shape (per ggml.h): [n_embd_v, n_head, n_batch_q, ne3]
        // = (d_pad, n_head, n_pos, n_batch). Slice ne[0] back to d_head.
        if (pad > 0) {
            KQV = ggml_view_4d(ctx, KQV, d_head, n_head, n_pos, n_batch,
                KQV->nb[1], KQV->nb[2], KQV->nb[3], 0);
            KQV = ggml_cont(ctx, KQV);
        }
        KQV = ggml_reshape_3d(ctx, KQV, d_head * n_head, n_pos, n_batch);
    } else {
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);                  // (d_head, n_pos, n_head, n_batch)
        K = ggml_permute(ctx, K, 0, 2, 1, 3);                  // same
        V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));  // (n_pos, d_head, n_head, n_batch)
        ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);            // (n_pos, n_pos, n_head, n_batch)
        KQ = ggml_soft_max_ext(ctx, KQ, /*mask=*/nullptr, kq_scale, 0.0f);
        KQV = ggml_mul_mat(ctx, V, KQ);                        // (d_head, n_pos, n_head, n_batch)
        KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);              // (d_head, n_head, n_pos, n_batch)
        KQV = ggml_cont(ctx, KQV);
        KQV = ggml_reshape_3d(ctx, KQV, d_head * n_head, n_pos, n_batch);
    }

    cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.o_w, KQV), layer.o_b);
    cur = ggml_add(ctx, cur, residual);
    residual = cur;

    cur = build_layernorm(ctx, cur, layer.ln2_w, layer.ln2_b, ln_eps);
    cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.up_w, cur), layer.up_b);
    cur = ggml_gelu(ctx, cur);
    cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.down_w, cur), layer.down_b);

    cur = ggml_add(ctx, cur, residual);
    return cur;
}

} // anonymous namespace

struct TextEncoder::State {
    qwen3_tts::GGUFLoader loader;

    ggml_backend_t        backend     = nullptr;
    ggml_backend_t        backend_cpu = nullptr;
    ggml_backend_buffer_t weights_buf = nullptr;
    ggml_context *        weights_ctx = nullptr;
    ggml_backend_sched_t  sched       = nullptr;

    ggml_tensor * token_embd    = nullptr; // (H, vocab)
    ggml_tensor * position_embd = nullptr; // (H, max_pos)
    ggml_tensor * final_ln_w    = nullptr;
    ggml_tensor * final_ln_b    = nullptr;
    ggml_tensor * head_w        = nullptr; // (H, projection_size)
    ggml_tensor * head_b        = nullptr;

    std::vector<Block> blocks;

    ~State() {
        if (sched) ggml_backend_sched_free(sched);
        if (weights_buf) ggml_backend_buffer_free(weights_buf);
        if (weights_ctx) ggml_free(weights_ctx);
        if (backend_cpu && backend_cpu != backend) ggml_backend_free(backend_cpu);
        if (backend) qwen3_tts::release_preferred_backend(backend);
    }
};

TextEncoder::TextEncoder() = default;

TextEncoder::~TextEncoder() {
    close();
}

void TextEncoder::close() {
    if (state_) {
        delete state_;
        state_ = nullptr;
    }
}

bool TextEncoder::load(const std::string & gguf_path) {
    close();
    state_ = new State{};

    if (!state_->loader.open(gguf_path)) {
        error_msg_ = state_->loader.get_error();
        delete state_;
        state_ = nullptr;
        return false;
    }

    config_.hidden_size             = state_->loader.get_u32(KV_HIDDEN, 1152);
    config_.intermediate_size       = state_->loader.get_u32(KV_INTER,  4304);
    config_.num_attention_heads     = state_->loader.get_u32(KV_HEADS,  16);
    config_.num_hidden_layers       = state_->loader.get_u32(KV_LAYERS, 27);
    config_.vocab_size              = state_->loader.get_u32(KV_VOCAB,  256000);
    config_.max_position_embeddings = state_->loader.get_u32(KV_MAX_POS, 64);
    config_.projection_size         = state_->loader.get_u32(KV_PROJ,   config_.hidden_size);
    config_.layer_norm_eps          = state_->loader.get_f32(KV_LN_EPS, 1e-6f);

    if (config_.hidden_size <= 0 || config_.num_hidden_layers <= 0 || config_.vocab_size <= 0) {
        error_msg_ = "GGUF metadata missing required text config keys";
        delete state_;
        state_ = nullptr;
        return false;
    }

    state_->backend = qwen3_tts::init_preferred_backend("siglip2-text", &error_msg_);
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

    state_->token_embd    = dup_meta(state_->weights_ctx, meta, "t.token_embd.weight");
    state_->position_embd = dup_meta(state_->weights_ctx, meta, "t.position_embd.weight");
    state_->final_ln_w    = dup_meta(state_->weights_ctx, meta, "t.final_ln.weight");
    state_->final_ln_b    = dup_meta(state_->weights_ctx, meta, "t.final_ln.bias");
    state_->head_w        = dup_meta(state_->weights_ctx, meta, "t.head.weight");
    state_->head_b        = dup_meta(state_->weights_ctx, meta, "t.head.bias");
    if (!add(state_->token_embd,    "t.token_embd.weight") ||
        !add(state_->position_embd, "t.position_embd.weight") ||
        !add(state_->final_ln_w,    "t.final_ln.weight") ||
        !add(state_->final_ln_b,    "t.final_ln.bias") ||
        !add(state_->head_w,        "t.head.weight") ||
        !add(state_->head_b,        "t.head.bias")) {
        delete state_;
        state_ = nullptr;
        return false;
    }

    state_->blocks.resize(config_.num_hidden_layers);
    for (int il = 0; il < config_.num_hidden_layers; ++il) {
        Block & b = state_->blocks[il];
        struct entry { ggml_tensor ** dst; const char * suffix; };
        entry entries[] = {
            {&b.ln1_w,  "ln1.weight"},      {&b.ln1_b,  "ln1.bias"},
            {&b.qkv_w,  "attn_qkv.weight"}, {&b.qkv_b,  "attn_qkv.bias"},
            {&b.o_w,    "attn_o.weight"},   {&b.o_b,    "attn_o.bias"},
            {&b.ln2_w,  "ln2.weight"},      {&b.ln2_b,  "ln2.bias"},
            {&b.up_w,   "ffn_up.weight"},   {&b.up_b,   "ffn_up.bias"},
            {&b.down_w, "ffn_down.weight"}, {&b.down_b, "ffn_down.bias"},
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
            max_nodes, false, true);
        if (!state_->sched) {
            error_msg_ = "ggml_backend_sched_new failed";
            delete state_;
            state_ = nullptr;
            return false;
        }
    }

    return true;
}

bool TextEncoder::encode(
    const int32_t *      token_ids,
    int                  n_tokens,
    const int32_t *      attention_mask,
    std::vector<float> & out_embedding) {
    if (!state_) {
        error_msg_ = "TextEncoder not loaded";
        return false;
    }
    if (n_tokens <= 0 || n_tokens > config_.max_position_embeddings) {
        char buf[128];
        snprintf(buf, sizeof(buf), "n_tokens must be in (0, %d], got %d",
            config_.max_position_embeddings, n_tokens);
        error_msg_ = buf;
        return false;
    }

    const int   H        = config_.hidden_size;
    const int   n_head   = config_.num_attention_heads;
    const int   d_head   = H / n_head;
    const int   n_pos    = n_tokens;
    const float kq_scale = 1.0f / std::sqrt((float)d_head);
    const int   proj     = config_.projection_size;

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

    // Inputs
    ggml_tensor * tok_inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_pos);
    ggml_set_name(tok_inp, "token_ids");
    ggml_set_input(tok_inp);

    // Position ids: 0..n_pos-1 (compile-time known, we feed at run time anyway).
    ggml_tensor * pos_inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_inp, "position_ids");
    ggml_set_input(pos_inp);

    // Attention mask (n_pos x n_pos), f32, 0 or -inf.
    ggml_tensor * mask_inp = nullptr;
    if (attention_mask) {
        mask_inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_pos, n_pos);
        ggml_set_name(mask_inp, "attn_mask");
        ggml_set_input(mask_inp);
    }

    // FA path uses a F16-cast of the mask reused across all layers.
    static const bool use_fa = std::getenv("SIGLIP2_DISABLE_FA") == nullptr;
    ggml_tensor * fa_mask_f16 = nullptr;
    if (use_fa && mask_inp) {
        fa_mask_f16 = ggml_cast(ctx, mask_inp, GGML_TYPE_F16);
    }

    // Token + position embedding lookups.
    ggml_tensor * x = ggml_get_rows(ctx, state_->token_embd, tok_inp);    // (H, n_pos)
    ggml_tensor * p = ggml_get_rows(ctx, state_->position_embd, pos_inp); // (H, n_pos)
    x = ggml_add(ctx, x, p);

    for (int il = 0; il < config_.num_hidden_layers; ++il) {
        x = build_block(ctx, state_->blocks[il], x, mask_inp, fa_mask_f16, n_pos, d_head, n_head,
                        config_.layer_norm_eps, kq_scale, use_fa);
    }

    x = build_layernorm(ctx, x, state_->final_ln_w, state_->final_ln_b, config_.layer_norm_eps);

    // Pool last position: x has ne=[H, n_pos]; ggml_get_rows indexes along ne[1].
    ggml_tensor * pool_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(pool_idx, "pool_idx");
    ggml_set_input(pool_idx);
    ggml_tensor * pooled = ggml_get_rows(ctx, ggml_cont(ctx, x), pool_idx); // ne=[H, 1]

    // Linear projection head: head_w stored ne=[H, proj_size]; mul_mat -> ne=[proj_size, 1]
    ggml_tensor * proj_out = ggml_add(ctx, ggml_mul_mat(ctx, state_->head_w, pooled), state_->head_b);
    ggml_set_name(proj_out, "text_embed");
    ggml_set_output(proj_out);
    ggml_build_forward_expand(gf, proj_out);

    if (!ggml_backend_sched_alloc_graph(state_->sched, gf)) {
        ggml_free(ctx);
        error_msg_ = "ggml_backend_sched_alloc_graph failed";
        return false;
    }

    ggml_backend_tensor_set(tok_inp, token_ids, 0, sizeof(int32_t) * n_pos);
    {
        std::vector<int32_t> pos(n_pos);
        for (int i = 0; i < n_pos; ++i) pos[i] = i;
        ggml_backend_tensor_set(pos_inp, pos.data(), 0, sizeof(int32_t) * n_pos);
    }
    if (mask_inp) {
        std::vector<float> mask((size_t)n_pos * n_pos, 0.0f);
        const float neg_inf = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < n_pos; ++k) {
            if (attention_mask[k] == 0) {
                for (int q = 0; q < n_pos; ++q) {
                    // Mask layout: ne[0]=n_pos_k innermost, ne[1]=n_pos_q outer.
                    mask[(size_t)q * n_pos + k] = neg_inf;
                }
            }
        }
        ggml_backend_tensor_set(mask_inp, mask.data(), 0, sizeof(float) * mask.size());
    }
    {
        const int32_t idx = (int32_t)(n_pos - 1);
        ggml_backend_tensor_set(pool_idx, &idx, 0, sizeof(int32_t));
    }

    enum ggml_status st = ggml_backend_sched_graph_compute(state_->sched, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(state_->sched);
        ggml_free(ctx);
        error_msg_ = std::string("graph compute failed status=") + std::to_string((int)st);
        return false;
    }

    out_embedding.assign((size_t)proj, 0.0f);
    ggml_backend_tensor_get(proj_out, out_embedding.data(), 0, sizeof(float) * (size_t)proj);

    ggml_backend_sched_reset(state_->sched);
    ggml_free(ctx);
    return true;
}

bool TextEncoder::encode_batch(
    const int32_t *      token_ids,
    int                  n_tokens,
    int                  n_batch,
    const int32_t *      attention_mask,
    std::vector<float> & out_embeddings) {
    if (!state_) {
        error_msg_ = "TextEncoder not loaded";
        return false;
    }
    if (n_batch <= 0) {
        error_msg_ = "n_batch must be > 0";
        return false;
    }
    // Single-prompt: dispatch to encode() so megakernel hooks fire.
    if (n_batch == 1) {
        return encode(token_ids, n_tokens, attention_mask, out_embeddings);
    }
    if (n_tokens <= 0 || n_tokens > config_.max_position_embeddings) {
        char buf[128];
        snprintf(buf, sizeof(buf), "n_tokens must be in (0, %d], got %d",
            config_.max_position_embeddings, n_tokens);
        error_msg_ = buf;
        return false;
    }
    // attention_mask in batched mode would require constructing a (n_pos, n_pos, 1, n_batch)
    // F32 mask and casting to F16 for FA. The live deployment matches HF — no
    // mask passed — so for now we reject the masked path and document.
    if (attention_mask != nullptr) {
        error_msg_ = "encode_batch with attention_mask is not implemented (HF path doesn't use one)";
        return false;
    }

    static const bool timing = std::getenv("SIGLIP2_TIME_ENCODE") != nullptr;
    auto tnow = []{ return std::chrono::steady_clock::now(); };
    auto tms  = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t0 = tnow();

    const int   H        = config_.hidden_size;
    const int   n_head   = config_.num_attention_heads;
    const int   d_head   = H / n_head;
    const int   n_pos    = n_tokens;
    const float kq_scale = 1.0f / std::sqrt((float)d_head);
    const int   proj     = config_.projection_size;

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
    auto t_init = tnow();

    // ggml_get_rows asserts source->ne[2] == idx->ne[1] (batched-table semantic),
    // which doesn't fit our shared-vocab/shared-pos-table lookups. Flatten the
    // indices to 1D, then reshape the result to 3D.
    ggml_tensor * tok_inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)n_pos * n_batch);
    ggml_set_name(tok_inp, "token_ids");
    ggml_set_input(tok_inp);

    ggml_tensor * pos_inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)n_pos * n_batch);
    ggml_set_name(pos_inp, "position_ids");
    ggml_set_input(pos_inp);

    static const bool use_fa = std::getenv("SIGLIP2_DISABLE_FA") == nullptr;

    // tok_emb / pos_emb: (H, n_pos*n_batch) → reshape_3d → (H, n_pos, n_batch)
    ggml_tensor * x = ggml_get_rows(ctx, state_->token_embd, tok_inp);
    ggml_tensor * p = ggml_get_rows(ctx, state_->position_embd, pos_inp);
    x = ggml_reshape_3d(ctx, x, H, n_pos, n_batch);
    p = ggml_reshape_3d(ctx, p, H, n_pos, n_batch);
    x = ggml_add(ctx, x, p);

    for (int il = 0; il < config_.num_hidden_layers; ++il) {
        x = build_block_batched(ctx, state_->blocks[il], x, /*fa_mask_f16=*/nullptr,
                                n_pos, n_batch, d_head, n_head,
                                config_.layer_norm_eps, kq_scale, use_fa);
    }

    x = build_layernorm(ctx, x, state_->final_ln_w, state_->final_ln_b, config_.layer_norm_eps);

    // Pool last position per batch. pool_idx is (1, n_batch); ggml_get_rows on
    // a 3D tensor with a 2D index yields (H, 1, n_batch) — each batch picks
    // its own row from its own slice along ne[2].
    ggml_tensor * pool_idx = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_batch);
    ggml_set_name(pool_idx, "pool_idx");
    ggml_set_input(pool_idx);
    ggml_tensor * pooled = ggml_get_rows(ctx, ggml_cont(ctx, x), pool_idx);
    pooled = ggml_reshape_2d(ctx, pooled, H, n_batch);

    ggml_tensor * proj_out = ggml_add(ctx, ggml_mul_mat(ctx, state_->head_w, pooled), state_->head_b);
    ggml_set_name(proj_out, "text_embed_batch");
    ggml_set_output(proj_out);
    ggml_build_forward_expand(gf, proj_out);
    auto t_build = tnow();

    if (!ggml_backend_sched_alloc_graph(state_->sched, gf)) {
        ggml_free(ctx);
        error_msg_ = "ggml_backend_sched_alloc_graph failed";
        return false;
    }
    auto t_alloc = tnow();

    ggml_backend_tensor_set(tok_inp, token_ids, 0,
                            sizeof(int32_t) * (size_t)n_pos * (size_t)n_batch);
    {
        // pos_inp is 1D (n_pos*n_batch); each batch's slice is 0..n_pos-1.
        std::vector<int32_t> pos((size_t)n_pos * (size_t)n_batch);
        for (int b = 0; b < n_batch; ++b) {
            for (int i = 0; i < n_pos; ++i) {
                pos[(size_t)b * (size_t)n_pos + (size_t)i] = i;
            }
        }
        ggml_backend_tensor_set(pos_inp, pos.data(), 0,
                                sizeof(int32_t) * (size_t)n_pos * (size_t)n_batch);
    }
    {
        std::vector<int32_t> idx(n_batch, (int32_t)(n_pos - 1));
        ggml_backend_tensor_set(pool_idx, idx.data(), 0, sizeof(int32_t) * (size_t)n_batch);
    }
    auto t_inputs = tnow();

    enum ggml_status st = ggml_backend_sched_graph_compute(state_->sched, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(state_->sched);
        ggml_free(ctx);
        error_msg_ = std::string("graph compute failed status=") + std::to_string((int)st);
        return false;
    }
    auto t_compute = tnow();

    out_embeddings.assign((size_t)proj * (size_t)n_batch, 0.0f);
    ggml_backend_tensor_get(proj_out, out_embeddings.data(), 0,
                            sizeof(float) * (size_t)proj * (size_t)n_batch);
    auto t_get = tnow();

    ggml_backend_sched_reset(state_->sched);
    ggml_free(ctx);
    auto t_done = tnow();

    if (timing) {
        std::fprintf(stderr,
            "[encode_batch n_batch=%d n_pos=%d] init=%.2f build=%.2f alloc=%.2f inputs=%.2f compute=%.2f get=%.2f reset=%.2f total=%.2f ms\n",
            n_batch, n_pos,
            tms(t0, t_init), tms(t_init, t_build), tms(t_build, t_alloc),
            tms(t_alloc, t_inputs), tms(t_inputs, t_compute),
            tms(t_compute, t_get), tms(t_get, t_done), tms(t0, t_done));
    }
    return true;
}

} // namespace siglip2
