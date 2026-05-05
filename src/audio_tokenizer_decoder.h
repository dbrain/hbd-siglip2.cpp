#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>
#include <memory>

namespace qwen3_tts {

// Audio tokenizer decoder (vocoder) configuration
struct audio_decoder_config {
    int32_t sample_rate = 24000;
    int32_t n_codebooks = 16;           // Total codebooks (1 first + 15 rest)
    int32_t codebook_size = 2048;       // Entries per codebook
    int32_t codebook_dim = 256;         // Embedding dimension per codebook
    int32_t latent_dim = 1024;          // Latent dimension after VQ
    int32_t hidden_dim = 512;           // Pre-transformer hidden dimension
    int32_t n_pre_tfm_layers = 8;       // Pre-transformer layers
    int32_t n_heads = 16;               // Attention heads in pre-transformer
    int32_t ffn_dim = 1024;             // FFN intermediate dimension
    int32_t decoder_dim = 1536;         // Initial decoder dimension
    int32_t upsample_rates[4] = {8, 5, 4, 3};  // Total: 480x upsampling
    // per-decoder-block output channels, populated at load time from the
    // transposed conv weights. used to size streaming-decode ring buffers.
    int32_t dec_out_channels[4] = {0, 0, 0, 0};
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
};

// Pre-transformer layer weights
struct pre_tfm_layer {
    // Attention
    struct ggml_tensor * attn_norm_w = nullptr;
    struct ggml_tensor * attn_q_w = nullptr;
    struct ggml_tensor * attn_k_w = nullptr;
    struct ggml_tensor * attn_v_w = nullptr;
    struct ggml_tensor * attn_output_w = nullptr;
    struct ggml_tensor * attn_scale = nullptr;  // layer_scale for attention
    
    // FFN (SwiGLU)
    struct ggml_tensor * ffn_norm_w = nullptr;
    struct ggml_tensor * ffn_gate_w = nullptr;
    struct ggml_tensor * ffn_up_w = nullptr;
    struct ggml_tensor * ffn_down_w = nullptr;
    struct ggml_tensor * ffn_scale = nullptr;   // layer_scale for FFN
};

// Residual block weights (Snake + Conv + Snake + Conv)
struct residual_block {
    int dilation = 1;  // Dilation for conv1: [1, 3, 9] for res[0], res[1], res[2]
    struct ggml_tensor * act1_alpha = nullptr;
    struct ggml_tensor * act1_beta = nullptr;
    struct ggml_tensor * conv1_w = nullptr;
    struct ggml_tensor * conv1_b = nullptr;
    struct ggml_tensor * act2_alpha = nullptr;
    struct ggml_tensor * act2_beta = nullptr;
    struct ggml_tensor * conv2_w = nullptr;
    struct ggml_tensor * conv2_b = nullptr;
};

// Decoder block weights (Snake + ConvTranspose + Residual blocks)
struct decoder_block {
    // Snake activation before conv transpose
    struct ggml_tensor * snake_alpha = nullptr;
    struct ggml_tensor * snake_beta = nullptr;
    
    // Transposed convolution for upsampling
    struct ggml_tensor * conv_t_w = nullptr;
    struct ggml_tensor * conv_t_b = nullptr;
    
    // Residual blocks (3 per decoder block)
    residual_block res[3];
};

// Upsample block weights (ConvNeXt-style)
struct upsample_block {
    struct ggml_tensor * conv_w = nullptr;
    struct ggml_tensor * conv_b = nullptr;
    struct ggml_tensor * dwconv_w = nullptr;
    struct ggml_tensor * dwconv_b = nullptr;
    struct ggml_tensor * norm_w = nullptr;
    struct ggml_tensor * norm_b = nullptr;
    struct ggml_tensor * pwconv1_w = nullptr;
    struct ggml_tensor * pwconv1_b = nullptr;
    struct ggml_tensor * pwconv2_w = nullptr;
    struct ggml_tensor * pwconv2_b = nullptr;
    struct ggml_tensor * gamma = nullptr;
};

// Audio tokenizer decoder model weights
struct audio_decoder_model {
    audio_decoder_config config;
    
    // VQ codebooks
    // vq_first: 1 codebook for first code
    struct ggml_tensor * vq_first_input_proj = nullptr;   // [1, 512, 256]
    struct ggml_tensor * vq_first_output_proj = nullptr;  // [1, 256, 512]
    struct ggml_tensor * vq_first_codebook = nullptr;     // [256, 2048] embedding_sum
    struct ggml_tensor * vq_first_usage = nullptr;        // [2048] cluster_usage
    
    // vq_rest: 15 codebooks for remaining codes
    struct ggml_tensor * vq_rest_input_proj = nullptr;    // [1, 512, 256]
    struct ggml_tensor * vq_rest_output_proj = nullptr;   // [1, 256, 512]
    struct ggml_tensor * vq_rest_codebook[15] = {nullptr}; // [256, 2048] embedding_sum each
    struct ggml_tensor * vq_rest_usage[15] = {nullptr};   // [2048] cluster_usage each
    
    // Upsample blocks (2 ConvNeXt-style blocks)
    upsample_block upsample[2];
    
    // Pre-transformer
    struct ggml_tensor * pre_tfm_input_proj_w = nullptr;  // [1024, 512]
    struct ggml_tensor * pre_tfm_input_proj_b = nullptr;
    pre_tfm_layer pre_tfm_layers[8];
    struct ggml_tensor * pre_tfm_norm_w = nullptr;        // Final RMSNorm
    struct ggml_tensor * pre_tfm_output_proj_w = nullptr; // [512, 1024]
    struct ggml_tensor * pre_tfm_output_proj_b = nullptr;
    
    // Pre-conv: [3, 512, 1024]
    struct ggml_tensor * pre_conv_w = nullptr;
    struct ggml_tensor * pre_conv_b = nullptr;
    
    // Decoder blocks
    // Block 0: Initial conv [7, 1024, 1536]
    struct ggml_tensor * dec0_conv_w = nullptr;
    struct ggml_tensor * dec0_conv_b = nullptr;
    
    // Blocks 1-4: Snake + ConvTranspose + 3 residual blocks
    decoder_block dec_blocks[4];
    
    // Block 5: Final snake activation
    struct ggml_tensor * dec5_snake_alpha = nullptr;
    struct ggml_tensor * dec5_snake_beta = nullptr;
    
    // Block 6: Output conv [7, 96, 1]
    struct ggml_tensor * dec6_conv_w = nullptr;
    struct ggml_tensor * dec6_conv_b = nullptr;
    
    // GGML context for tensor metadata
    struct ggml_context * ctx = nullptr;
    
    // Backend buffer for weights
    ggml_backend_buffer_t buffer = nullptr;
    
    // Tensor name to tensor mapping
    std::map<std::string, struct ggml_tensor *> tensors;
};

// Compute state for decoder
struct audio_decoder_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
};

// Audio tokenizer decoder (vocoder) class
// Decodes discrete audio codes to waveform
class AudioTokenizerDecoder {
public:
    AudioTokenizerDecoder();
    ~AudioTokenizerDecoder();

    // Set abort callback checked before each graph compute (thread-safe)
    void set_abort_callback(ggml_abort_callback callback, void * data);
    bool is_aborted() const;

    // Load model from GGUF file (tokenizer model)
    bool load_model(const std::string & model_path);

    // Release all model/runtime resources
    void unload_model();
    
    // Decode audio codes to waveform
    // codes: audio codes [n_frames, n_codebooks] as int32_t (row-major)
    // n_frames: number of frames
    // Returns: audio samples normalized to [-1, 1] at 24kHz
    bool decode(const int32_t * codes, int32_t n_frames,
                std::vector<float> & samples);

    // Stateful streaming decode. Appends PCM for the given chunk of codes
    // to `samples`, carrying KV-cache and causal-conv tail state across
    // calls. Call stream_reset() between independent utterances.
    // On the first call (n_past == 0) with the same codes this produces
    // bit-identical PCM to decode(); subsequent chunks continue the stream.
    bool stream_decode(const int32_t * codes, int32_t n_frames,
                       std::vector<float> & samples);

    // Reset streaming state. Must be called before starting a new utterance.
    void stream_reset();

    // Persistent host-side snapshot of streaming state. Used to skip ICL
    // ref-codes warmup re-decode when the same reference has already been
    // decoded once and the resulting vocoder state is cached at a higher
    // layer (Qwen3TTS::icl_cache_, keyed by ref_codes hash).
    //
    // Only the persistent fields are captured: n_past_, host tail rings,
    // conv_transpose overlap buffers, and the per-layer KV host caches.
    // The transient stream_tails_/stream_conv_ts_/stream_kvs_ vectors are
    // rebuilt by build_graph() on every stream_decode() call, so they
    // need not be saved.
    struct stream_state_snapshot {
        int32_t n_past = 0;
        std::map<std::string, std::vector<float>> tail_rings;
        std::map<std::string, std::vector<float>> conv_t_overlap_hosts;
        std::vector<std::vector<float>> past_k_hosts;
        std::vector<std::vector<float>> past_v_hosts;
    };

    // Capture current streaming state (typically right after a stream_decode
    // call that consumed the ref_codes warmup). Cheap — just copies host
    // vectors. Caller should take this immediately after the warmup
    // decode and *before* any further stream_decode() that would advance
    // the state.
    void capture_stream_state(stream_state_snapshot & out) const;

    // Restore streaming state from a snapshot. Marks the decoder as in
    // streaming mode so the next stream_decode() picks up where the
    // captured state left off. Bypasses the ref-codes warmup entirely.
    void restore_stream_state(const stream_state_snapshot & snap);

    const audio_decoder_config & get_config() const { return model_.config; }

    // VRAM inventory dump — weights buffer + sched per-backend reservations.
    void log_vram_breakdown(const char * label) const;

    const std::string & get_error() const { return error_msg_; }
    
private:
    // Build computation graph for decoding. n_past is the KV-cache /
    // causal-conv history length from prior calls; 0 for one-shot decode.
    // When n_past > 0, the graph additionally reads per-layer past_K/past_V
    // inputs and applies an n_past-aware attention mask + positions.
    struct ggml_cgraph * build_graph(int32_t n_frames, int32_t n_past = 0);
    
    // Apply Snake activation: x + (1/alpha) * sin^2(alpha * x)
    struct ggml_tensor * apply_snake(struct ggml_context * ctx,
                                      struct ggml_tensor * x,
                                      struct ggml_tensor * alpha,
                                      struct ggml_tensor * beta);
    
    // Apply RMSNorm
    struct ggml_tensor * apply_rms_norm(struct ggml_context * ctx,
                                         struct ggml_tensor * x,
                                         struct ggml_tensor * w,
                                         float eps);
    
    // Apply pre-transformer layer. n_past is the prior KV-cache length
    // (0 in one-shot mode); mask uses n_past as the diagonal offset.
    // layer_idx identifies the layer for streaming KV-cache tensor names.
    struct ggml_tensor * apply_pre_tfm_layer(struct ggml_context * ctx,
                                              struct ggml_tensor * x,
                                              const pre_tfm_layer & layer,
                                              int32_t n_frames,
                                              int32_t n_past,
                                              int32_t layer_idx,
                                              struct ggml_tensor * positions);
    
    // Apply upsample block (ConvNeXt-style). block_idx is used to build
    // unique tail tensor names.
    struct ggml_tensor * apply_upsample_block(struct ggml_context * ctx,
                                               struct ggml_tensor * x,
                                               const upsample_block & block,
                                               int block_idx);

    // Apply residual block. tail_name_prefix forms the tail input name
    // (e.g. "tail_dec1_res0_conv1").
    struct ggml_tensor * apply_residual_block(struct ggml_context * ctx,
                                               struct ggml_tensor * x,
                                               const residual_block & block,
                                               const char * tail_name_prefix);

    // Apply decoder block (Snake + ConvTranspose + Residuals)
    struct ggml_tensor * apply_decoder_block(struct ggml_context * ctx,
                                              struct ggml_tensor * x,
                                              const decoder_block & block,
                                              int upsample_rate,
                                              int block_idx);
    
    audio_decoder_model model_;
    audio_decoder_state state_;
    std::string error_msg_;
    ggml_abort_callback abort_cb_ = nullptr;
    void * abort_data_ = nullptr;
    
    // Temporary storage for codes input
    std::vector<int32_t> codes_buf_;

    // Names of all causal-conv tail input tensors created during build_graph.
    // decode() iterates these to zero-fill for one-shot mode; streaming mode
    // sets them from the ring buffers below.
    std::vector<std::string> tail_names_;

    // Streaming mode flag consulted by build_graph to gate the KV-cache
    // and next_* output extensions.
    bool streaming_mode_ = false;

    // Per-call streaming tail metadata, rebuilt by build_graph. next_node
    // points to the ggml_cont tensor that emits the last L frames and must
    // be passed to ggml_build_forward_expand so the scheduler keeps it live.
    struct stream_tail {
        std::string in_name;
        std::string out_name;
        int L = 0;
        int channels = 0;
        struct ggml_tensor * next_node = nullptr;
    };
    std::vector<stream_tail> stream_tails_;

    // Persistent per-tail host ring buffers, keyed by in_name.
    // Preserved across calls; cleared by stream_reset().
    std::map<std::string, std::vector<float>> tail_rings_;

    // Per-call streaming conv_transpose overlap metadata per decoder block.
    // The raw transpose output is (n_in+1)*s samples long; one-shot trims s
    // samples off each side. Under streaming we hold back the right s
    // samples and overlap-add them onto the first s samples of the next
    // chunk's raw transpose output.
    struct stream_conv_t {
        std::string in_name;      // conv_t_overlap_in_{block}
        std::string out_name;     // next_conv_t_overlap_{block}
        int stride = 0;           // == upsample_rate == s
        int channels = 0;
        struct ggml_tensor * next_node = nullptr;
    };
    std::vector<stream_conv_t> stream_conv_ts_;

    // Persistent per-block raw-right-tail host buffers, keyed by in_name.
    std::map<std::string, std::vector<float>> conv_t_overlap_hosts_;

    // Per-call streaming KV metadata per attention layer, rebuilt by build_graph.
    struct stream_kv {
        std::string past_k_in, past_v_in;
        std::string next_k_out, next_v_out;
        struct ggml_tensor * next_k_node = nullptr;
        struct ggml_tensor * next_v_node = nullptr;
    };
    std::vector<stream_kv> stream_kvs_;

    // Persistent per-layer KV host buffers. Size = n_past_ * n_heads * head_dim.
    std::vector<std::vector<float>> past_k_hosts_;
    std::vector<std::vector<float>> past_v_hosts_;

    int32_t n_past_ = 0;  // current KV / tail history length

    // Create a causal-conv tail input tensor, concat it with x along dim 0,
    // register the input in tail_names_, and (in streaming mode) also emit
    // the trailing L frames as a named output so the driver can roll it
    // into the next call's input tail.
    struct ggml_tensor * make_causal_tail(struct ggml_context * ctx,
                                           struct ggml_tensor * x,
                                           int L, int channels,
                                           const char * in_name);
};

// Free model resources
void free_audio_decoder_model(audio_decoder_model & model);

} // namespace qwen3_tts
