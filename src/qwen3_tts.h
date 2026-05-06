#pragma once

#include "text_tokenizer.h"
#include "tts_transformer.h"
#include "audio_tokenizer_encoder.h"
#include "audio_codec_encoder.h"
#include "audio_tokenizer_decoder.h"

#include <string>
#include <vector>
#include <functional>
#include <cstdint>
#include <map>

namespace qwen3_tts {

// TTS generation parameters
struct tts_params {
    // Maximum number of audio tokens to generate
    int32_t max_audio_tokens = 2048;
    
    // Temperature for sampling (0 = greedy)
    float temperature = 0.9f;
    
    // Top-p sampling
    float top_p = 1.0f;
    
    // Top-k sampling (0 = disabled)
    int32_t top_k = 50;
    
    // Number of threads
    int32_t n_threads = 4;
    
    // Print progress during generation
    bool print_progress = false;
    
    // Print timing information
    bool print_timing = true;
    
    // Repetition penalty for CB0 token generation (HuggingFace style)
    float repetition_penalty = 1.05f;

    // Language ID for codec (2050=en, 2069=ru, 2055=zh, 2058=ja, 2064=ko, 2053=de, 2061=fr, 2054=es)
    int32_t language_id = 2050;

    // Voice steering instruction (e.g. "speak slowly in a calm tone")
    std::string instructions;

    // Reference text for ICL voice cloning (when set, enables ICL mode instead of x-vector)
    std::string ref_text;

    // Sampling seed. < 0 means leave RNG as-is (non-deterministic); >= 0 reseeds
    // the transformer's RNG so runs are reproducible.
    int64_t seed = -1;
};

// TTS generation result
struct tts_result {
    // Generated audio samples (24kHz, mono)
    std::vector<float> audio;
    
    // Sample rate
    int32_t sample_rate = 24000;
    
    // Success flag
    bool success = false;
    
    // Error message if failed
    std::string error_msg;
    
    // Token counts (real, not approximated)
    int32_t n_text_tokens = 0;      // user-content text tokens (maps to openai usage.input_tokens)
    int32_t n_prefill_tokens = 0;   // total positions the transformer prefilled
                                    // (text + instruct + ref_text + ref_codes + framing)
    int32_t n_audio_tokens = 0;     // codec frames produced by the transformer

    // Timing info (in milliseconds)
    int64_t t_load_ms = 0;
    int64_t t_tokenize_ms = 0;
    int64_t t_encode_ms = 0;        // speaker encoder (voice cloning)
    int64_t t_generate_ms = 0;      // full transformer generate() wall time
    int64_t t_prefill_ms = 0;       // subset of t_generate_ms: build_prefill + forward_prefill
    int64_t t_decode_ms = 0;        // vocoder decode
    int64_t t_total_ms = 0;

    // Process memory snapshots (bytes)
    uint64_t mem_rss_start_bytes = 0;
    uint64_t mem_rss_end_bytes = 0;
    uint64_t mem_rss_peak_bytes = 0;
    uint64_t mem_phys_start_bytes = 0;
    uint64_t mem_phys_end_bytes = 0;
    uint64_t mem_phys_peak_bytes = 0;

    // Cache keys this synth used. Both are 0 for non-cacheable synths
    // (e.g. raw-audio path with a per-call speaker embedding). The
    // server uses these to persist the now-populated caches alongside
    // the voice bundle. See save_voice_warmup.
    uint64_t prefill_cache_key = 0;
    uint64_t ref_codes_hash    = 0;
};

// Progress callback type
using tts_progress_callback_t = std::function<void(int tokens_generated, int max_tokens)>;

// Streaming decode options. When batch_size > 0, the transformer emits
// audio codes in frame batches that are decoded live via the audio
// decoder's streaming path, and each decoded PCM batch is forwarded to
// `on_pcm`. A final flush drains any trailing partial batch. The
// aggregate PCM is also accumulated into `tts_result::audio` for parity
// with the non-streaming path, but consumers that only care about wire
// bytes can ignore it.
struct streaming_opts {
    int32_t batch_size = 0;
    // Batch size for the FIRST emit only. If 0, falls back to batch_size.
    // Set to a small value (e.g. 1-5) to minimise time-to-first-byte while
    // letting subsequent batches stay large enough to keep throughput up.
    int32_t first_batch_size = 0;
    std::function<bool(const float * pcm, size_t n_samples)> on_pcm;
};

// Main TTS class that orchestrates the full pipeline
class Qwen3TTS {
public:
    Qwen3TTS();
    ~Qwen3TTS();
    
    // Load all models from directory (auto-detects q8_0 vs f16)
    // model_dir should contain: transformer.gguf, tokenizer.gguf, vocoder.gguf
    bool load_models(const std::string & model_dir);

    // Load models from explicit file paths
    // tts_model_path: path to the TTS GGUF (tokenizer + transformer + encoder)
    // vocoder_model_path: path to the vocoder GGUF (if empty, looks in same directory)
    // speaker_encoder_model_path: optional separate GGUF that contains spk_enc.* tensors
    //   and qwen3-tts.speaker_encoder.* metadata. When set, the speaker encoder is loaded
    //   from this path instead of from tts_model_path. Use case: VoiceDesign GGUF (no
    //   spk_enc) + Base GGUF (has spk_enc) — gives one resident model with both
    //   description-driven voices AND audio-reference cloning.
    bool load_model_files(const std::string & tts_model_path,
                          const std::string & vocoder_model_path = "",
                          const std::string & speaker_encoder_model_path = "");
    
    // Generate speech from text
    // text: input text to synthesize
    // params: generation parameters
    tts_result synthesize(const std::string & text,
                          const tts_params & params = tts_params());
    
    // Generate speech with voice cloning
    // text: input text to synthesize
    // reference_audio: path to reference audio file (WAV, 24kHz)
    // params: generation parameters
    tts_result synthesize_with_voice(const std::string & text,
                                      const std::string & reference_audio,
                                      const tts_params & params = tts_params());
    
    // Generate speech with voice cloning from samples
    // text: input text to synthesize
    // ref_samples: reference audio samples (24kHz, mono, normalized to [-1, 1])
    // n_ref_samples: number of reference samples
    // params: generation parameters
    tts_result synthesize_with_voice(const std::string & text,
                                      const float * ref_samples, int32_t n_ref_samples,
                                      const tts_params & params = tts_params());

    // Extract speaker embedding from reference audio file (without synthesis)
    // reference_audio: path to reference audio file (WAV)
    // embedding: output vector of 1024 float32 values
    bool extract_speaker_embedding(const std::string & reference_audio,
                                    std::vector<float> & embedding);

    // Encode audio to discrete speech codes for ICL voice cloning
    // samples: audio samples (24kHz, mono, normalized to [-1, 1])
    // n_samples: number of samples
    // codes: output vector of codes (n_frames * 16 interleaved)
    // n_frames: output number of frames
    bool encode_speech_codes(const float * samples, int32_t n_samples,
                              std::vector<int32_t> & codes, int32_t & n_frames);

    // Generate speech with pre-extracted speaker embedding
    // text: input text to synthesize
    // embedding: pre-extracted speaker embedding (1024 float32 values)
    // embedding_size: number of elements in embedding (must be 1024)
    // params: generation parameters
    tts_result synthesize_with_embedding(const std::string & text,
                                          const float * embedding, int32_t embedding_size,
                                          const tts_params & params = tts_params(),
                                          const int32_t * ref_codes = nullptr,
                                          int32_t n_ref_frames = 0,
                                          const streaming_opts * stream = nullptr);

    // Streaming overload for non-voice-clone synthesis. See streaming_opts.
    tts_result synthesize(const std::string & text,
                          const tts_params & params,
                          const streaming_opts * stream);

    // Query model info
    int32_t get_hidden_size() const;
    int32_t get_sample_rate() const;
    const std::string & get_model_type() const;
    const std::vector<std::string> & get_speaker_names() const;
    const std::vector<int32_t> & get_speaker_ids() const;
    bool has_speaker_encoder() const;

    // Look up speaker token ID by name (-1 if not found)
    int32_t get_speaker_id(const std::string & name) const;

    // Get speaker embedding for a built-in speaker (from codec_embd at speaker token ID)
    bool get_speaker_embedding(const std::string & name, std::vector<float> & embedding);

    // Set progress callback
    void set_progress_callback(tts_progress_callback_t callback);

    // Set abort callback on all loaded component backends (thread-safe).
    // The callback is stored and automatically re-applied after lazy load/reload.
    void set_abort_callback(ggml_abort_callback callback, void * data);

    // Get error message
    const std::string & get_error() const { return error_msg_; }

    // Check if models are loaded
    bool is_loaded() const { return models_loaded_; }

    // Drop any cached ICL warmup vocoder states (host-side). Call after a
    // change that would invalidate cached state (e.g. vocoder model swap).
    void clear_icl_cache();

    // Release the speaker encoder + codec encoder VRAM. They are needed
    // only at voice-registration time (extract speaker embedding +
    // ref_codes from raw audio); once a voice's bundle is on disk the
    // synthesis path uses cached embedding/codes and never touches the
    // encoders. The server triggers this at the end of /v1/audio/voices
    // POST handlers; the encoders lazily reload on the next register.
    // Saves ~250 MiB of permanently-resident VRAM after the first
    // register call.
    void unload_encoders();

    // -- Persistent voice caches (cold-path elimination) ------------------
    //
    // After a successful synth, three host-side caches are populated for
    // the voice: talker prefill KV snapshot, icl_codec_section embedding
    // bytes, and vocoder ICL warmup state. Together these eliminate
    // ~700 ms of cold-path TTFA on Q4 voice_clone. Persisting them to
    // disk lets a server that frequently unloads/reloads (e.g. shared
    // GPU) skip the cold-path tax across restarts.
    //
    // Lifecycle: server calls save_voice_warmup(voice_id, <keys>, path)
    // after first successful synth for the voice, and load_voice_warmup
    // on startup before the model serves traffic. Format is self-describing;
    // load is best-effort — wrong model_id, version, or shape returns
    // false silently and the next synth rebuilds the caches.
    bool save_voice_warmup(const std::string & voice_id,
                           uint64_t prefill_cache_key,
                           uint64_t ref_codes_hash,
                           const std::string & path,
                           const std::string & model_id);

    // Load previously-saved voice warmup state for the given path. The
    // file's stored prefill_cache_key + ref_codes_hash become the keys
    // under which the in-memory caches are seeded — subsequent synths
    // that produce matching keys (deterministic from the same inputs)
    // hit the cache without rebuilding. Returns true if at least one
    // section was loaded; missing/stale/mismatched files return false.
    bool load_voice_warmup(const std::string & path,
                           const std::string & model_id);

    // Release all model GPU/CPU buffers, schedulers, and backends. After
    // this call, is_loaded() returns false until reload_model() (or
    // load_model_files()) is called. Synthesis attempts will fail until
    // the model is reloaded. Host-side caches (icl_cache_, voices map in
    // server.cpp) are NOT cleared — they remain valid across unload/reload
    // of the same model.
    void unload_model();

    // Reload the model files using paths captured by the most recent
    // load_model_files() / load_models() call. Cheap no-op if already
    // loaded. Returns false (with get_error()) on failure.
    bool reload_model();

    // Cache model paths without touching GPU. Used by lazy-load mode in
    // server.cpp: paths get registered up front so the first request (or
    // /reload, or idle-unload reload) can drive load_model_files() — but
    // startup itself doesn't touch the GPU. Safe to call before any load.
    void set_model_paths(const std::string & tts_model_path,
                         const std::string & vocoder_model_path,
                         const std::string & speaker_encoder_model_path);

private:
    tts_result synthesize_internal(const std::string & text,
                                   const float * speaker_embedding,
                                   const tts_params & params,
                                   tts_result & result,
                                   const int32_t * ref_codes = nullptr,
                                   int32_t n_ref_frames = 0,
                                   const streaming_opts * stream = nullptr);

    // Lazy-load the speaker encoder and verify its embedding dim matches
    // the talker's hidden_size. Sets error_msg_ and returns false on
    // failure. Idempotent.
    bool ensure_encoder_loaded();

    bool is_aborted() const { return abort_cb_ && abort_cb_(abort_data_); }
    
    TextTokenizer tokenizer_;
    TTSTransformer transformer_;
    AudioTokenizerEncoder audio_encoder_;
    AudioCodecEncoder codec_encoder_;
    AudioTokenizerDecoder audio_decoder_;
    
    bool models_loaded_ = false;
    bool encoder_loaded_ = false;
    bool codec_encoder_loaded_ = false;
    bool transformer_loaded_ = false;
    bool decoder_loaded_ = false;
    bool low_mem_mode_ = false;
    std::string error_msg_;
    std::string tts_model_path_;
    std::string decoder_model_path_;
    std::string speaker_encoder_model_path_;
    tts_progress_callback_t progress_callback_;
    ggml_abort_callback abort_cb_ = nullptr;
    void * abort_data_ = nullptr;

    // ICL warmup vocoder-state cache. Keyed by FNV-1a hash of the ref_codes
    // int32 byte stream. On a hit we restore the streaming decoder state
    // captured after the first warmup pass and skip the (~700–1200 ms)
    // re-decode of identical ref codes on every cloned-voice synth.
    // Bounded growth: each entry is a few MB host RAM; in practice keyed
    // by registered voices, so size == n_voices.
    std::map<uint64_t, AudioTokenizerDecoder::stream_state_snapshot> icl_cache_;
};

// Utility: Load audio file (WAV format)
bool load_audio_file(const std::string & path, std::vector<float> & samples, 
                     int & sample_rate);

// Utility: Save audio file (WAV format)
bool save_audio_file(const std::string & path, const std::vector<float> & samples,
                     int sample_rate);

} // namespace qwen3_tts
