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
    // Fused Q/K/V: one wider mul_mat replaces three Q/K/V projections.
    // Weight is (H, 3*H), bias is (3*H,). Q/K/V are strided views afterwards.
    ggml_tensor * qkv_w  = nullptr; ggml_tensor * qkv_b  = nullptr;
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

// SigLIP2's d_head=72 doesn't divide by 16, so the CUDA MMA / WMMA flash-attn
// kernels can't accept it directly (their tile templates require D%16==0).
// We work around that by zero-padding Q/K/V's d-axis to the next multiple of 16
// before FA, and slicing the result back to the real d_head before the output
// projection. Zero-padded contributions are mathematically null:
//   Q·K  → padded q_d * padded k_d = 0·0 = 0      (no change in attention scores)
//   V    → padded v_d in result = 0               (sliced away before o_w)
// On Ampere this routes attention to the tensor-core MMA kernel (case 80) and
// keeps the existing 0.999 parity floor.
constexpr int FA_TC_ALIGN = 16;

inline int round_up_d(int d) { return (d + FA_TC_ALIGN - 1) & ~(FA_TC_ALIGN - 1); }

// Pads d-axis (if d_head not multiple of 16) on the F32 side, then casts K/V
// to F16 for FA. ggml's CUDA pad op only accepts F32 sources, so we do the
// pad before the cast.
ggml_tensor * fa_attn_pad_slice(
    ggml_context * ctx,
    ggml_tensor *  Q,                   // (d_head, ?, n_head) F32, post-permute (non-contig view ok)
    ggml_tensor *  K,                   // (d_head, ?, n_head) F32, post-permute (non-contig view ok)
    ggml_tensor *  V,                   // (d_head, ?, n_head) F32, post-permute (non-contig view ok)
    ggml_tensor *  fa_mask_f16,         // optional mask, F16
    int            d_head,
    int            n_head,
    int            n_pos_q,
    float          kq_scale) {
    const int d_pad = round_up_d(d_head);
    if (d_pad != d_head) {
        const int pad = d_pad - d_head;
        // ggml_cuda's pad op requires F32 + contiguous source; ggml_cont covers both.
        Q = ggml_pad(ctx, ggml_cont(ctx, Q), pad, 0, 0, 0);
        K = ggml_pad(ctx, ggml_cont(ctx, K), pad, 0, 0, 0);
        V = ggml_pad(ctx, ggml_cont(ctx, V), pad, 0, 0, 0);
    }
    ggml_tensor * K_f16 = ggml_cast(ctx, K, GGML_TYPE_F16);
    ggml_tensor * V_f16 = ggml_cast(ctx, V, GGML_TYPE_F16);
    ggml_tensor * KQV = ggml_flash_attn_ext(ctx, Q, K_f16, V_f16, fa_mask_f16, kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(KQV, GGML_PREC_F32);
    // FA result: ne=[d_pad, n_head, n_pos_q, 1]
    if (d_pad != d_head) {
        // Slice ne[0] back to real d_head and rebuild contiguous (H, n_pos_q).
        KQV = ggml_view_3d(ctx, KQV, d_head, n_head, n_pos_q, KQV->nb[1], KQV->nb[2], 0);
        KQV = ggml_cont(ctx, KQV);
    }
    return ggml_reshape_2d(ctx, KQV, d_head * n_head, n_pos_q);
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

    // Fused QKV mul_mat (saves 2 mul_mats + 2 bias-adds per layer × 27 layers).
    const int    H  = d_head * n_head;
    const size_t es = sizeof(float);
    ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, layer.qkv_w, cur), layer.qkv_b);
    ggml_tensor * Q = ggml_view_3d(ctx, qkv, d_head, n_head, n_pos,
        /*nb1=*/d_head * es, /*nb2=*/3 * H * es, /*offset=*/0 * H * es);
    ggml_tensor * K = ggml_view_3d(ctx, qkv, d_head, n_head, n_pos,
        /*nb1=*/d_head * es, /*nb2=*/3 * H * es, /*offset=*/1 * H * es);
    ggml_tensor * V = ggml_view_3d(ctx, qkv, d_head, n_head, n_pos,
        /*nb1=*/d_head * es, /*nb2=*/3 * H * es, /*offset=*/2 * H * es);

    ggml_tensor * KQV;
    if (use_fa) {
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);                 // (d_head, n_pos, n_head)
        K = ggml_permute(ctx, K, 0, 2, 1, 3);                 // (d_head, n_pos, n_head)
        V = ggml_permute(ctx, V, 0, 2, 1, 3);                 // (d_head, n_pos, n_head) — NOT transposed
        // fa_attn_pad_slice does the F16 cast + pad-when-needed internally.
        KQV = fa_attn_pad_slice(ctx, Q, K, V, /*mask*/ nullptr, d_head, n_head, n_pos, kq_scale);
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
        KQV = fa_attn_pad_slice(ctx, Q, K, V, /*mask*/ nullptr, d_head, n_head, /*n_pos_q=*/1, kq_scale);
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

// Phase B Path 2: cache (ctx, gf, gallocr, named tensors) per (n_patches_h, n_patches_w, pooling).
//
// Same approach as TextEncoder — bypass ggml_backend_sched, use a per-entry
// gallocr + ggml_backend_graph_compute. Stable tensor pointers across calls
// unlock ggml-cuda's CUDA-graph warmup path.
struct VisionGraphCacheEntry {
    int     n_patches_h = 0;
    int     n_patches_w = 0;
    Pooling pooling     = Pooling::MEAN;

    std::vector<uint8_t> arena;
    ggml_context *       ctx     = nullptr;
    ggml_cgraph *        gf      = nullptr;
    ggml_gallocr_t       gallocr = nullptr;

    ggml_tensor * inp    = nullptr; // pixel_values input
    ggml_tensor * pooled = nullptr; // output

    VisionGraphCacheEntry() = default;
    VisionGraphCacheEntry(const VisionGraphCacheEntry &) = delete;
    VisionGraphCacheEntry & operator=(const VisionGraphCacheEntry &) = delete;
    VisionGraphCacheEntry(VisionGraphCacheEntry && o) noexcept { *this = std::move(o); }
    VisionGraphCacheEntry & operator=(VisionGraphCacheEntry && o) noexcept {
        if (this != &o) {
            if (gallocr) ggml_gallocr_free(gallocr);
            if (ctx) ggml_free(ctx);
            n_patches_h = o.n_patches_h;
            n_patches_w = o.n_patches_w;
            pooling     = o.pooling;
            arena       = std::move(o.arena);
            ctx         = o.ctx;     o.ctx = nullptr;
            gf          = o.gf;      o.gf  = nullptr;
            gallocr     = o.gallocr; o.gallocr = nullptr;
            inp         = o.inp;     o.inp = nullptr;
            pooled      = o.pooled;  o.pooled = nullptr;
        }
        return *this;
    }
    ~VisionGraphCacheEntry() {
        if (gallocr) ggml_gallocr_free(gallocr);
        if (ctx) ggml_free(ctx);
    }
};

struct VisionEncoder::State {
    qwen3_tts::GGUFLoader loader;

    ggml_backend_t        backend     = nullptr;
    ggml_backend_buffer_t weights_buf = nullptr;
    ggml_context *        weights_ctx = nullptr;

    // Weight tensors (in weights_ctx)
    ggml_tensor * patch_embd_w = nullptr;
    ggml_tensor * patch_embd_b = nullptr;
    ggml_tensor * pos_embd     = nullptr;
    ggml_tensor * post_ln_w    = nullptr;
    ggml_tensor * post_ln_b    = nullptr;

    std::vector<Block> blocks;
    ProbeHead          head;

    // NaFlex hits a finite set of (h, w) shapes per max_num_patches; 4 entries
    // covers the binary-search outcomes for max_num_patches=729 (square + a
    // few aspect-clamped variants). LRU evicts further shapes.
    static constexpr size_t kCacheCap = 4;
    std::vector<VisionGraphCacheEntry> graph_cache;

    ~State() {
        graph_cache.clear();
        if (weights_buf) {
            ggml_backend_buffer_free(weights_buf);
        }
        if (weights_ctx) {
            ggml_free(weights_ctx);
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

    static const bool use_fa = std::getenv("SIGLIP2_DISABLE_FA") == nullptr;

    VisionGraphCacheEntry * pe = nullptr;
    for (auto & ce : state_->graph_cache) {
        if (ce.n_patches_h == n_patches_h && ce.n_patches_w == n_patches_w && ce.pooling == pooling) {
            pe = &ce;
            break;
        }
    }
    bool was_miss = pe == nullptr;
    if (was_miss) {
        if (state_->graph_cache.size() >= State::kCacheCap) {
            state_->graph_cache.erase(state_->graph_cache.begin());
        }
        state_->graph_cache.emplace_back();
        VisionGraphCacheEntry & e = state_->graph_cache.back();
        e.n_patches_h = n_patches_h;
        e.n_patches_w = n_patches_w;
        e.pooling     = pooling;

        const int    max_nodes = 2048;
        const size_t arena_sz  =
            ggml_tensor_overhead() * 4096 +
            ggml_graph_overhead_custom(max_nodes, false);
        e.arena.assign(arena_sz, 0);
        struct ggml_init_params gp = {
            /*.mem_size   =*/ e.arena.size(),
            /*.mem_buffer =*/ e.arena.data(),
            /*.no_alloc   =*/ true,
        };
        e.ctx = ggml_init(gp);
        if (!e.ctx) {
            state_->graph_cache.pop_back();
            error_msg_ = "ggml_init for graph_ctx failed";
            return false;
        }
        e.gf = ggml_new_graph_custom(e.ctx, max_nodes, false);

        e.inp = ggml_new_tensor_2d(e.ctx, GGML_TYPE_F32, feat, n_pos);
        ggml_set_name(e.inp, "pixel_values");
        ggml_set_input(e.inp);

        ggml_tensor * x = ggml_mul_mat(e.ctx, state_->patch_embd_w, e.inp);
        x = ggml_add(e.ctx, x, state_->patch_embd_b);

        ggml_tensor * pos_embd = state_->pos_embd;
        if (n_patches_h != n_per_side || n_patches_w != n_per_side) {
            pos_embd = ggml_reshape_3d(e.ctx, pos_embd, H, n_per_side, n_per_side);
            pos_embd = ggml_permute(e.ctx, pos_embd, 2, 0, 1, 3);
            pos_embd = ggml_interpolate(
                e.ctx, pos_embd,
                n_patches_w, n_patches_h, H, 1,
                GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS);
            pos_embd = ggml_permute(e.ctx, pos_embd, 1, 2, 0, 3);
            pos_embd = ggml_cont_2d(e.ctx, pos_embd, H, n_pos);
        }
        x = ggml_add(e.ctx, x, pos_embd);

        for (int il = 0; il < config_.num_hidden_layers; ++il) {
            x = build_block(e.ctx, state_->blocks[il], x, n_pos, d_head, n_head, config_.layer_norm_eps, kq_scale, use_fa);
        }

        x = build_layernorm(e.ctx, x, state_->post_ln_w, state_->post_ln_b, config_.layer_norm_eps);

        if (pooling == Pooling::MEAN) {
            e.pooled = ggml_cont(e.ctx, ggml_transpose(e.ctx, x));
            e.pooled = ggml_mean(e.ctx, e.pooled);
        } else {
            e.pooled = build_probe_head(
                e.ctx, state_->head, x,
                n_pos, H, d_head, n_head, config_.layer_norm_eps, kq_scale, use_fa);
        }
        ggml_set_name(e.pooled, "pooled");
        ggml_set_output(e.pooled);
        ggml_build_forward_expand(e.gf, e.pooled);

        e.gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(state_->backend));
        if (!e.gallocr || !ggml_gallocr_reserve(e.gallocr, e.gf)) {
            state_->graph_cache.pop_back();
            error_msg_ = "ggml_gallocr_reserve failed";
            return false;
        }

        pe = &e;
    }
    VisionGraphCacheEntry & ce = *pe;

    if (was_miss) {
        if (!ggml_gallocr_alloc_graph(ce.gallocr, ce.gf)) {
            error_msg_ = "ggml_gallocr_alloc_graph failed";
            return false;
        }
    }

    ggml_backend_tensor_set(ce.inp, pixel_values, 0, sizeof(float) * (size_t)feat * n_pos);

    enum ggml_status st = ggml_backend_graph_compute(state_->backend, ce.gf);
    if (st != GGML_STATUS_SUCCESS) {
        error_msg_ = std::string("graph compute failed status=") + std::to_string((int)st);
        return false;
    }

    out_embedding.assign((size_t)H, 0.0f);
    ggml_backend_tensor_get(ce.pooled, out_embedding.data(), 0, sizeof(float) * (size_t)H);

    return true;
}

} // namespace siglip2
