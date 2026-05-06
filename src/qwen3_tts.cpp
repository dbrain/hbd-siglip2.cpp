#include "qwen3_tts.h"
#include "gguf_loader.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <type_traits>

#ifdef __APPLE__
#include <mach/mach.h>
#else
#include <sys/resource.h>
#endif

namespace qwen3_tts {

static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct process_memory_snapshot {
    uint64_t rss_bytes = 0;
    uint64_t phys_footprint_bytes = 0;
};

static bool get_process_memory_snapshot(process_memory_snapshot & out) {
#ifdef __APPLE__
    mach_task_basic_info_data_t basic_info = {};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&basic_info), &basic_count) != KERN_SUCCESS) {
        return false;
    }
    out.rss_bytes = (uint64_t) basic_info.resident_size;

    task_vm_info_data_t vm_info = {};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&vm_info), &vm_count) == KERN_SUCCESS) {
        out.phys_footprint_bytes = (uint64_t) vm_info.phys_footprint;
    } else {
        out.phys_footprint_bytes = out.rss_bytes;
    }
    return true;
#else
    struct rusage usage = {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return false;
    }
    out.rss_bytes = (uint64_t) usage.ru_maxrss * 1024ULL;
    out.phys_footprint_bytes = out.rss_bytes;
    return true;
#endif
}

static std::string format_bytes(uint64_t bytes) {
    static const char * units[] = { "B", "KB", "MB", "GB", "TB" };
    double val = (double) bytes;
    int unit = 0;
    while (val >= 1024.0 && unit < 4) {
        val /= 1024.0;
        ++unit;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f %s", val, units[unit]);
    return std::string(buf);
}

static void log_memory_usage(const char * label) {
    process_memory_snapshot mem;
    if (!get_process_memory_snapshot(mem)) {
        fprintf(stderr, "  [mem] %-24s unavailable\n", label);
        return;
    }
    fprintf(stderr, "  [mem] %-24s rss=%s  phys=%s\n",
            label, format_bytes(mem.rss_bytes).c_str(),
            format_bytes(mem.phys_footprint_bytes).c_str());
}

static void resample_linear(const float * input, int input_len, int input_rate,
                            std::vector<float> & output, int output_rate) {
    double ratio = (double)input_rate / output_rate;
    int output_len = (int)((double)input_len / ratio);
    output.resize(output_len);
    
    for (int i = 0; i < output_len; ++i) {
        double src_idx = i * ratio;
        int idx0 = (int)src_idx;
        int idx1 = idx0 + 1;
        double frac = src_idx - idx0;
        
        if (idx1 >= input_len) {
            output[i] = input[input_len - 1];
        } else {
            output[i] = (float)((1.0 - frac) * input[idx0] + frac * input[idx1]);
        }
    }
}

Qwen3TTS::Qwen3TTS() = default;

Qwen3TTS::~Qwen3TTS() = default;

bool Qwen3TTS::load_models(const std::string & model_dir) {
    // discover talker (any *.gguf not matching tokenizer) and vocoder (*tokenizer*.gguf).
    // prefer q8_0 over f16 for talker.
    namespace fs = std::filesystem;
    std::string tts_path, vocoder_path;
    std::string tts_q8, tts_f16, tts_other;
    std::error_code ec;
    for (const auto & entry : fs::directory_iterator(model_dir, ec)) {
        if (ec || !entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (name.size() < 5 || name.substr(name.size() - 5) != ".gguf") continue;
        std::string lower = name;
        for (auto & c : lower) c = (char)std::tolower((unsigned char)c);
        if (lower.find("tokenizer") != std::string::npos) {
            vocoder_path = entry.path().string();
        } else if (lower.find("q8_0") != std::string::npos) {
            tts_q8 = entry.path().string();
        } else if (lower.find("f16") != std::string::npos || lower.find("fp16") != std::string::npos) {
            tts_f16 = entry.path().string();
        } else {
            tts_other = entry.path().string();
        }
    }
    tts_path = !tts_q8.empty() ? tts_q8 : (!tts_f16.empty() ? tts_f16 : tts_other);
    if (tts_path.empty() || vocoder_path.empty()) {
        error_msg_ = "could not find talker and tokenizer ggufs in " + model_dir;
        return false;
    }
    return load_model_files(tts_path, vocoder_path);
}

bool Qwen3TTS::load_model_files(const std::string & tts_path,
                                 const std::string & vocoder_path,
                                 const std::string & speaker_encoder_path) {
    int64_t t_start = get_time_ms();
    log_memory_usage("load/start");

    transformer_.unload_model();
    audio_decoder_.unload_model();
    encoder_loaded_ = false;
    transformer_loaded_ = false;
    decoder_loaded_ = false;

    tts_model_path_ = tts_path;
    speaker_encoder_model_path_ = speaker_encoder_path;

    // derive vocoder path from same directory if not specified
    if (vocoder_path.empty()) {
        auto slash = tts_path.rfind('/');
        std::string dir = (slash != std::string::npos) ? tts_path.substr(0, slash) : ".";
        decoder_model_path_ = dir + "/qwen3-tts-tokenizer-f16.gguf";
    } else {
        decoder_model_path_ = vocoder_path;
    }

    if (!speaker_encoder_model_path_.empty()) {
        fprintf(stderr, "  Speaker encoder source: %s (separate GGUF)\n",
                speaker_encoder_model_path_.c_str());
    }

    const char * low_mem_env = std::getenv("QWEN3_TTS_LOW_MEM");
    low_mem_mode_ = low_mem_env && low_mem_env[0] != '\0' && low_mem_env[0] != '0';
    if (low_mem_mode_) {
        fprintf(stderr, "  Low-memory mode enabled (lazy decoder + component unloads)\n");
    }

    // Load TTS model (contains text tokenizer + transformer for generation)
    fprintf(stderr, "Loading TTS model from %s...\n", tts_model_path_.c_str());

    // Load text tokenizer from TTS model
    int64_t t_tokenizer_start = get_time_ms();
    {
        GGUFLoader loader;
        if (!loader.open(tts_model_path_)) {
            error_msg_ = "Failed to open TTS model: " + loader.get_error();
            return false;
        }

        if (!tokenizer_.load_from_gguf(loader.get_ctx())) {
            error_msg_ = "Failed to load text tokenizer: " + tokenizer_.get_error();
            return false;
        }
        fprintf(stderr, "  Text tokenizer loaded: vocab_size=%d (%lld ms)\n",
                tokenizer_.get_config().vocab_size,
                (long long)(get_time_ms() - t_tokenizer_start));
    }
    log_memory_usage("load/after-tokenizer");

    // Speaker encoder is loaded lazily on first voice cloning request.
    fprintf(stderr, "  Speaker encoder: deferred (lazy load)\n");

    // Load TTS transformer from TTS model
    int64_t t_transformer_start = get_time_ms();
    if (!transformer_.load_model(tts_model_path_)) {
        error_msg_ = "Failed to load TTS transformer: " + transformer_.get_error();
        fprintf(stderr, "  ERROR: %s\n", error_msg_.c_str());
        return false;
    }
    transformer_loaded_ = true;
    fprintf(stderr, "  TTS transformer loaded: hidden_size=%d, n_layers=%d (%lld ms)\n",
            transformer_.get_config().hidden_size, transformer_.get_config().n_layers,
            (long long)(get_time_ms() - t_transformer_start));
    log_memory_usage("load/after-transformer");

    if (!low_mem_mode_) {
        // Load vocoder (audio decoder) from tokenizer model
        fprintf(stderr, "Loading vocoder from %s...\n", decoder_model_path_.c_str());
        int64_t t_decoder_start = get_time_ms();
        if (!audio_decoder_.load_model(decoder_model_path_)) {
            error_msg_ = "Failed to load vocoder: " + audio_decoder_.get_error();
            fprintf(stderr, "  ERROR: %s\n", error_msg_.c_str());
            return false;
        }
        decoder_loaded_ = true;
        fprintf(stderr, "  Vocoder loaded: sample_rate=%d, n_codebooks=%d (%lld ms)\n",
                audio_decoder_.get_config().sample_rate, audio_decoder_.get_config().n_codebooks,
                (long long)(get_time_ms() - t_decoder_start));
        log_memory_usage("load/after-vocoder");
    } else {
        fprintf(stderr, "  Vocoder: deferred (lazy load)\n");
    }

    models_loaded_ = true;

    int64_t t_end = get_time_ms();
    fprintf(stderr, "All models loaded in %lld ms\n", (long long)(t_end - t_start));
    log_memory_usage("load/end");

    return true;
}

tts_result Qwen3TTS::synthesize(const std::string & text,
                                 const tts_params & params) {
    return synthesize(text, params, nullptr);
}

tts_result Qwen3TTS::synthesize(const std::string & text,
                                 const tts_params & params,
                                 const streaming_opts * stream) {
    tts_result result;

    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    // For basic synthesis without voice cloning, we use a zero speaker embedding
    // This will use the model's default voice characteristics
    std::vector<float> zero_embedding(transformer_.get_config().hidden_size, 0.0f);

    return synthesize_internal(text, zero_embedding.data(), params, result,
                               nullptr, 0, stream);
}

tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                            const std::string & reference_audio,
                                            const tts_params & params) {
    tts_result result;
    
    std::vector<float> ref_samples;
    int ref_sample_rate;
    if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
        result.error_msg = "Failed to load reference audio: " + reference_audio;
        return result;
    }
    
    const int target_rate = 24000;
    if (ref_sample_rate != target_rate) {
        fprintf(stderr, "Resampling audio from %d Hz to %d Hz...\n", ref_sample_rate, target_rate);
        std::vector<float> resampled;
        resample_linear(ref_samples.data(), (int)ref_samples.size(), ref_sample_rate, resampled, target_rate);
        ref_samples = std::move(resampled);
    }
    
    return synthesize_with_voice(text, ref_samples.data(), (int32_t)ref_samples.size(), params);
}

tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                            const float * ref_samples, int32_t n_ref_samples,
                                            const tts_params & params) {
    tts_result result;
    
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    if (!encoder_loaded_) {
        int64_t t_encoder_load_start = get_time_ms();
        if (!ensure_encoder_loaded()) {
            result.error_msg = error_msg_;
            return result;
        }
        if (params.print_timing) {
            fprintf(stderr, "  Speaker encoder lazy-loaded in %lld ms\n",
                    (long long)(get_time_ms() - t_encoder_load_start));
            log_memory_usage("voice/after-encoder-load");
        }
    }

    int64_t t_encode_start = get_time_ms();
    std::vector<float> speaker_embedding;

    if (!audio_encoder_.encode(ref_samples, n_ref_samples, speaker_embedding)) {
        result.error_msg = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return result;
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;
    
    if (params.print_progress) {
        fprintf(stderr, "Speaker embedding extracted: %zu floats\n", speaker_embedding.size());
    }

    // ICL mode: also encode reference audio to discrete speech codes
    if (!params.ref_text.empty()) {
        if (!codec_encoder_loaded_) {
            if (decoder_model_path_.empty()) {
                result.error_msg = "missing tokenizer model path for codec encoder";
                return result;
            }
            int64_t t0 = get_time_ms();
            if (!codec_encoder_.load_model(decoder_model_path_)) {
                result.error_msg = "failed to load codec encoder: " + codec_encoder_.get_error();
                return result;
            }
            codec_encoder_loaded_ = true;
            if (params.print_timing) {
                fprintf(stderr, "  Codec encoder lazy-loaded in %lld ms\n",
                        (long long)(get_time_ms() - t0));
            }
        }

        int64_t t0 = get_time_ms();
        std::vector<int32_t> ref_codes;
        int32_t n_ref_frames = 0;
        if (!codec_encoder_.encode(ref_samples, n_ref_samples, ref_codes, n_ref_frames)) {
            result.error_msg = "failed to encode reference audio: " + codec_encoder_.get_error();
            return result;
        }
        if (params.print_progress) {
            fprintf(stderr, "Reference audio encoded: %d frames x 16 codebooks (ICL mode)\n", n_ref_frames);
        }
        if (params.print_timing) {
            fprintf(stderr, "  Codec encode: %lld ms\n", (long long)(get_time_ms() - t0));
        }

        return synthesize_internal(text, speaker_embedding.data(), params, result,
                                   ref_codes.data(), n_ref_frames);
    }

    return synthesize_internal(text, speaker_embedding.data(), params, result);
}

bool Qwen3TTS::extract_speaker_embedding(const std::string & reference_audio,
                                           std::vector<float> & embedding) {
    if (!models_loaded_) {
        error_msg_ = "Models not loaded";
        return false;
    }

    std::vector<float> ref_samples;
    int ref_sample_rate;
    if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
        error_msg_ = "Failed to load reference audio: " + reference_audio;
        return false;
    }

    const int target_rate = 24000;
    if (ref_sample_rate != target_rate) {
        std::vector<float> resampled;
        resample_linear(ref_samples.data(), (int)ref_samples.size(), ref_sample_rate, resampled, target_rate);
        ref_samples = std::move(resampled);
    }

    if (!ensure_encoder_loaded()) {
        return false;
    }

    if (!audio_encoder_.encode(ref_samples.data(), (int32_t)ref_samples.size(), embedding)) {
        error_msg_ = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return false;
    }

    return true;
}

bool Qwen3TTS::encode_speech_codes(const float * samples, int32_t n_samples,
                                    std::vector<int32_t> & codes, int32_t & n_frames) {
    if (!models_loaded_) {
        error_msg_ = "Models not loaded";
        return false;
    }

    if (!codec_encoder_loaded_) {
        if (decoder_model_path_.empty()) {
            error_msg_ = "missing tokenizer model path for codec encoder";
            return false;
        }
        if (!codec_encoder_.load_model(decoder_model_path_)) {
            error_msg_ = "failed to load codec encoder: " + codec_encoder_.get_error();
            return false;
        }
        codec_encoder_loaded_ = true;
    }

    if (!codec_encoder_.encode(samples, n_samples, codes, n_frames)) {
        error_msg_ = "failed to encode speech codes: " + codec_encoder_.get_error();
        return false;
    }

    return true;
}

tts_result Qwen3TTS::synthesize_with_embedding(const std::string & text,
                                                 const float * embedding, int32_t embedding_size,
                                                 const tts_params & params,
                                                 const int32_t * ref_codes,
                                                 int32_t n_ref_frames,
                                                 const streaming_opts * stream) {
    tts_result result;

    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    if (!embedding) {
        result.error_msg = "Speaker embedding is null";
        return result;
    }

    const int32_t expected_size = transformer_.get_config().hidden_size;
    if (embedding_size != expected_size) {
        result.error_msg = "Invalid embedding size: expected " + std::to_string(expected_size)
                         + ", got " + std::to_string(embedding_size);
        return result;
    }

    return synthesize_internal(text, embedding, params, result, ref_codes, n_ref_frames, stream);
}

tts_result Qwen3TTS::synthesize_internal(const std::string & text,
                                          const float * speaker_embedding,
                                          const tts_params & params,
                                          tts_result & result,
                                          const int32_t * ref_codes,
                                          int32_t n_ref_frames,
                                          const streaming_opts * stream) {
    int64_t t_total_start = get_time_ms();
    auto sample_memory = [&](const char * stage) {
        process_memory_snapshot mem;
        if (!get_process_memory_snapshot(mem)) {
            return;
        }
        if (result.mem_rss_start_bytes == 0) {
            result.mem_rss_start_bytes = mem.rss_bytes;
            result.mem_phys_start_bytes = mem.phys_footprint_bytes;
        }
        result.mem_rss_end_bytes = mem.rss_bytes;
        result.mem_phys_end_bytes = mem.phys_footprint_bytes;
        if (mem.rss_bytes > result.mem_rss_peak_bytes) {
            result.mem_rss_peak_bytes = mem.rss_bytes;
        }
        if (mem.phys_footprint_bytes > result.mem_phys_peak_bytes) {
            result.mem_phys_peak_bytes = mem.phys_footprint_bytes;
        }
        if (params.print_timing) {
            fprintf(stderr, "  [mem] %-24s rss=%s  phys=%s\n",
                    stage,
                    format_bytes(mem.rss_bytes).c_str(),
                    format_bytes(mem.phys_footprint_bytes).c_str());
        }
    };
    sample_memory("synth/start");
    
    // Step 2: Tokenize input text and optional instruction prompt
    int64_t t_tokenize_start = get_time_ms();
    std::vector<int32_t> text_tokens = tokenizer_.encode_for_tts(text);
    std::vector<int32_t> instruct_tokens;
    if (!params.instructions.empty() && transformer_.get_config().model_size != "0b6") {
        instruct_tokens = tokenizer_.encode_instruct(params.instructions);
    }
    result.t_tokenize_ms = get_time_ms() - t_tokenize_start;
    result.n_text_tokens = (int32_t)text_tokens.size();
    sample_memory("synth/after-tokenize");

    if (text_tokens.empty()) {
        result.error_msg = "Failed to tokenize text";
        return result;
    }
    
    if (params.print_progress) {
        fprintf(stderr, "Text tokenized: %zu tokens\n", text_tokens.size());
        fprintf(stderr, "  Tokens: ");
        for (size_t i = 0; i < std::min(text_tokens.size(), (size_t)10); ++i) {
            fprintf(stderr, "%d ", text_tokens[i]);
        }
        if (text_tokens.size() > 10) fprintf(stderr, "...");
        fprintf(stderr, "\n");
    }
    
    // Step 3: Generate speech codes using TTS transformer
    int64_t t_generate_start = get_time_ms();
    if (!transformer_loaded_) {
        int64_t t_reload_start = get_time_ms();
        if (!transformer_.load_model(tts_model_path_)) {
            result.error_msg = "Failed to reload TTS transformer: " + transformer_.get_error();
            return result;
        }
        transformer_.set_abort_callback(abort_cb_, abort_data_);
        transformer_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Transformer reloaded in %lld ms\n",
                    (long long)(get_time_ms() - t_reload_start));
            sample_memory("synth/after-transformer-reload");
        }
    }
    transformer_.clear_kv_cache();
    transformer_.set_verbose(params.print_progress);
    if (params.seed >= 0) {
        transformer_.set_seed((uint64_t)params.seed);
    }

    // tokenize ref_text for ICL mode
    std::vector<int32_t> ref_text_tokens;
    if (ref_codes && n_ref_frames > 0 && !params.ref_text.empty()) {
        ref_text_tokens = tokenizer_.encode_for_tts(params.ref_text);
        // python _build_ref_text wraps as <|im_start|>assistant\n{text}<|im_end|>\n
        // and slices ref_id[:, 3:-2], yielding just the content tokens. our
        // encode_for_tts uses the longer assistant wrap with a trailing
        // <|im_start|>assistant\n (8 framing tokens), so we drop 3 prefix + 5
        // suffix tokens to land on the same content-only window. leaving the
        // boundary tokens in causes the talker to treat ref+new as separate
        // turns, producing hangs and ref-text interjection.
        if ((int)ref_text_tokens.size() > 8) {
            ref_text_tokens = std::vector<int32_t>(
                ref_text_tokens.begin() + 3,
                ref_text_tokens.end() - 5
            );
        } else {
            ref_text_tokens.clear();
        }
    }

    // streaming: install per-frame callback that batches codes and live-decodes
    // via the vocoder's streaming path. we must also ensure the decoder is
    // loaded before generate() so its stream_decode can be called from within
    // the callback. ICL warm-up (ref_codes) is fed as a discarded chunk below.
    const bool streaming = stream && stream->batch_size > 0 && stream->on_pcm;
    std::vector<int32_t> stream_buf;
    size_t stream_cb_count = 0;
    bool stream_cb_aborted = false;
    if (streaming) {
        if (!decoder_loaded_) {
            int64_t t_decoder_load_start = get_time_ms();
            if (decoder_model_path_.empty()) {
                result.error_msg = "Internal error: missing vocoder model path";
                return result;
            }
            if (!audio_decoder_.load_model(decoder_model_path_)) {
                result.error_msg = "Failed to load vocoder: " + audio_decoder_.get_error();
                return result;
            }
            audio_decoder_.set_abort_callback(abort_cb_, abort_data_);
            decoder_loaded_ = true;
            if (params.print_timing) {
                fprintf(stderr, "  Vocoder lazy-loaded in %lld ms\n",
                        (long long)(get_time_ms() - t_decoder_load_start));
                sample_memory("synth/after-vocoder-load-stream");
            }
        }
        audio_decoder_.stream_reset();
        const int n_cb = transformer_.get_config().n_codebooks;

        // ICL warm-up: feed ref_codes through the streaming decoder and
        // discard its PCM. Identical for the same voice every synth, so
        // cache the resulting vocoder state keyed by FNV-1a hash of the
        // ref_codes byte stream and restore on subsequent hits — saves
        // ~700-1200 ms TTFA on every cloned-voice synth.
        if (ref_codes && n_ref_frames > 0 && !params.ref_text.empty()) {
            // Hash matches build_prefill_graph's icl_codec_section_cache_ key
            // and result.ref_codes_hash exactly so all three caches keyed by
            // ref_codes content share one consistent identifier — important
            // for the persistent voice.warmup blob to round-trip correctly.
            const size_t n_bytes = (size_t) n_ref_frames * n_cb * sizeof(int32_t);
            uint64_t hash = 1469598103934665603ull; // FNV-1a 64 offset basis
            const uint8_t * p = reinterpret_cast<const uint8_t *>(ref_codes);
            for (size_t i = 0; i < n_bytes; ++i) {
                hash ^= p[i];
                hash *= 1099511628211ull;
            }
            int32_t mix[2] = { n_ref_frames, n_cb };
            const uint8_t * mp = reinterpret_cast<const uint8_t *>(mix);
            for (size_t i = 0; i < sizeof(mix); ++i) {
                hash ^= mp[i];
                hash *= 1099511628211ull;
            }

            int64_t t_warmup_start = get_time_ms();
            auto it = icl_cache_.find(hash);
            if (it != icl_cache_.end()) {
                audio_decoder_.restore_stream_state(it->second);
                if (params.print_timing) {
                    fprintf(stderr, "  ICL warmup: cache HIT (hash=%016llx, %d frames) — restored in %lld ms\n",
                            (unsigned long long) hash, n_ref_frames,
                            (long long)(get_time_ms() - t_warmup_start));
                }
            } else {
                std::vector<float> warmup_pcm;
                if (!audio_decoder_.stream_decode(ref_codes, n_ref_frames, warmup_pcm)) {
                    result.error_msg = "Failed to warm-up vocoder with ref codes: " + audio_decoder_.get_error();
                    return result;
                }
                AudioTokenizerDecoder::stream_state_snapshot snap;
                audio_decoder_.capture_stream_state(snap);
                icl_cache_.emplace(hash, std::move(snap));
                if (params.print_timing) {
                    fprintf(stderr, "  ICL warmup: cache MISS (hash=%016llx, %d frames) — decoded + cached in %lld ms\n",
                            (unsigned long long) hash, n_ref_frames,
                            (long long)(get_time_ms() - t_warmup_start));
                }
                // discard warmup_pcm — downstream only sees post-ref PCM
            }
        }

        stream_buf.reserve((size_t) stream->batch_size * n_cb);
        transformer_.set_frame_callback(
            [this, stream, &stream_buf, &stream_cb_count, &stream_cb_aborted, n_cb, &result]
            (int32_t /*frame_idx*/, const int32_t * frame_codes) -> bool {
                for (int c = 0; c < n_cb; ++c) stream_buf.push_back(frame_codes[c]);
                const int frames_buffered = (int) (stream_buf.size() / n_cb);
                // First emit can use a smaller batch to minimise TTFB; later
                // emits use the larger batch_size for throughput.
                const int target = (stream_cb_count == 0 && stream->first_batch_size > 0)
                                   ? stream->first_batch_size
                                   : stream->batch_size;
                if (frames_buffered >= target) {
                    std::vector<float> pcm;
                    if (!audio_decoder_.stream_decode(stream_buf.data(), frames_buffered, pcm)) {
                        stream_cb_aborted = true;
                        return false;
                    }
                    stream_buf.clear();
                    result.audio.insert(result.audio.end(), pcm.begin(), pcm.end());
                    stream_cb_count++;
                    if (!stream->on_pcm(pcm.data(), pcm.size())) {
                        stream_cb_aborted = true;
                        return false;
                    }
                }
                return true;
            });
    }

    // Voice-keyed prefill cache key. Hash everything that determines the
    // cacheable [prefix..end-of-ref_text] portion of the prefill assembly:
    // language, ICL flag, has_speaker, the speaker embedding bytes, the
    // instruct tokens, and the ref_text tokens. Anything past ref_text
    // (new_text, codec_bos, ref_codes) is per-request and excluded.
    uint64_t prefill_cache_key = 0;
    {
        const bool icl_mode    = (ref_codes && n_ref_frames > 0 && !ref_text_tokens.empty());
        const bool has_speaker = (speaker_embedding != nullptr);
        const int32_t hsize    = transformer_.get_config().hidden_size;
        uint64_t h = 1469598103934665603ull; // FNV-1a 64 offset basis
        auto add = [&h](const void * data, size_t n) {
            const uint8_t * p = static_cast<const uint8_t *>(data);
            for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
        };
        int32_t lang  = params.language_id;
        uint8_t flags = (uint8_t) ((has_speaker ? 1 : 0) | (icl_mode ? 2 : 0));
        add(&lang,  sizeof(lang));
        add(&flags, sizeof(flags));
        if (has_speaker) {
            add(speaker_embedding, (size_t) hsize * sizeof(float));
        }
        int32_t n_inst = (int32_t) instruct_tokens.size();
        add(&n_inst, sizeof(n_inst));
        if (n_inst > 0) {
            add(instruct_tokens.data(), (size_t) n_inst * sizeof(int32_t));
        }
        int32_t n_reft = (int32_t) ref_text_tokens.size();
        add(&n_reft, sizeof(n_reft));
        if (n_reft > 0) {
            add(ref_text_tokens.data(), (size_t) n_reft * sizeof(int32_t));
        }
        prefill_cache_key = h;
        if (prefill_cache_key == 0) prefill_cache_key = 1; // 0 reserved for "no cache"
    }
    result.prefill_cache_key = prefill_cache_key;

    // Also compute the ref_codes_hash that build_prefill_graph and the ICL
    // warmup path key into. Server uses both keys to drive disk persistence
    // of the now-populated caches via save_voice_warmup.
    if (ref_codes && n_ref_frames > 0) {
        const int32_t n_cb_for_hash = transformer_.get_config().n_codebooks;
        const size_t n_bytes = (size_t) n_ref_frames * n_cb_for_hash * sizeof(int32_t);
        uint64_t h = 1469598103934665603ull;
        const uint8_t * p = reinterpret_cast<const uint8_t *>(ref_codes);
        for (size_t i = 0; i < n_bytes; ++i) { h ^= p[i]; h *= 1099511628211ull; }
        int32_t mix[2] = { n_ref_frames, n_cb_for_hash };
        const uint8_t * mp = reinterpret_cast<const uint8_t *>(mix);
        for (size_t i = 0; i < sizeof(mix); ++i) { h ^= mp[i]; h *= 1099511628211ull; }
        result.ref_codes_hash = h;
    }

    std::vector<int32_t> speech_codes;
    if (!transformer_.generate(text_tokens.data(), (int32_t)text_tokens.size(),
                               speaker_embedding, params.max_audio_tokens, speech_codes,
                               params.language_id, params.repetition_penalty,
                               params.temperature, params.top_k,
                               instruct_tokens.empty() ? nullptr : instruct_tokens.data(),
                               (int32_t)instruct_tokens.size(),
                               ref_text_tokens.empty() ? nullptr : ref_text_tokens.data(),
                               (int32_t)ref_text_tokens.size(),
                               ref_codes, n_ref_frames,
                               prefill_cache_key)) {
        if (streaming) transformer_.set_frame_callback({});
        result.error_msg = "Failed to generate speech codes: " + transformer_.get_error();
        return result;
    }
    if (streaming) {
        transformer_.set_frame_callback({});
    }
    result.t_generate_ms = get_time_ms() - t_generate_start;
    result.n_prefill_tokens = transformer_.get_last_n_prefill_tokens();
    result.t_prefill_ms     = transformer_.get_last_prefill_ms();
    sample_memory("synth/after-generate");

    // QWEN3_TTS_LOG_SCHED=1 → one-shot vocoder VRAM breakdown alongside
    // the talker breakdown emitted from forward_prefill.
    static bool vocoder_vram_logged = false;
    if (!vocoder_vram_logged && std::getenv("QWEN3_TTS_LOG_SCHED")) {
        vocoder_vram_logged = true;
        if (decoder_loaded_) audio_decoder_.log_vram_breakdown("post-first-generate");
    }

    if (is_aborted()) {
        result.error_msg = "Aborted";
        return result;
    }

    int n_codebooks = transformer_.get_config().n_codebooks;
    int n_frames = (int)speech_codes.size() / n_codebooks;
    result.n_audio_tokens = n_frames;

    if (params.print_progress) {
        fprintf(stderr, "Speech codes generated: %d frames x %d codebooks\n", n_frames, n_codebooks);
    }
    
    if (n_frames == 0) {
        result.error_msg = "No speech codes generated";
        return result;
    }

    if (low_mem_mode_) {
        transformer_.unload_model();
        transformer_loaded_ = false;
        sample_memory("synth/after-transformer-unload");
    }
    
    // Step 4: Decode speech codes to waveform using vocoder
    int64_t t_decode_start = get_time_ms();

    // streaming path: frames were already decoded live in the frame callback.
    // flush any residual (< batch_size) frames now, then skip the one-shot
    // decode that follows.
    if (streaming) {
        const int n_cb = transformer_.get_config().n_codebooks;
        const int leftover = (int) (stream_buf.size() / n_cb);
        if (leftover > 0 && !stream_cb_aborted) {
            std::vector<float> pcm;
            if (!audio_decoder_.stream_decode(stream_buf.data(), leftover, pcm)) {
                result.error_msg = "Failed to flush streaming vocoder: " + audio_decoder_.get_error();
                return result;
            }
            stream_buf.clear();
            result.audio.insert(result.audio.end(), pcm.begin(), pcm.end());
            if (!stream->on_pcm(pcm.data(), pcm.size())) {
                stream_cb_aborted = true;
            }
        }
        result.t_decode_ms = get_time_ms() - t_decode_start;
        sample_memory("synth/after-stream-decode");
        if (params.print_progress) {
            fprintf(stderr, "Streaming: %zu batches dispatched, %zu total samples\n",
                    stream_cb_count, result.audio.size());
        }
        result.sample_rate = audio_decoder_.get_config().sample_rate;
        result.success = !stream_cb_aborted;
        if (stream_cb_aborted && result.error_msg.empty()) {
            result.error_msg = "Streaming consumer aborted";
        }
        result.t_total_ms = get_time_ms() - t_total_start;
        sample_memory("synth/end");
        return result;
    }

    if (!decoder_loaded_) {
        int64_t t_decoder_load_start = get_time_ms();
        if (decoder_model_path_.empty()) {
            result.error_msg = "Internal error: missing vocoder model path";
            return result;
        }
        if (!audio_decoder_.load_model(decoder_model_path_)) {
            result.error_msg = "Failed to load vocoder: " + audio_decoder_.get_error();
            return result;
        }
        audio_decoder_.set_abort_callback(abort_cb_, abort_data_);
        decoder_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Vocoder lazy-loaded in %lld ms\n",
                    (long long)(get_time_ms() - t_decoder_load_start));
            sample_memory("synth/after-vocoder-load");
        }
    }
    
    // ICL: previously we'd prepend ref_codes to the talker output so the
    // vocoder gets warm context (match qwen3_tts_model.py:616), then trim the
    // ref portion off the decoded wav. Two problems with that:
    //   (1) proportional trim was off because the decoder doesn't produce
    //       constant samples-per-frame, especially at silence boundaries — left
    //       multiple seconds of ref bleeding into the start of cloned output.
    //   (2) decoding ref alone (to measure exact length for trim) gives a
    //       different boundary state than decoding [ref + new] together, so
    //       even an "exact" trim by ref-alone-decode-length still leaks audible
    //       ref content from the join point.
    //
    // Cleanest fix: don't prepend ref at decode. Decode just the talker's new
    // codes directly. PyTorch hybrid testing (phase 6d C-noconcat) showed no
    // audible cold-start click in practice, so the warm-up motivation doesn't
    // hold up empirically on this codec.
    std::vector<int32_t> codes_for_decode;  // unused now; kept as marker for skipped trim path
    int32_t total_frames = n_frames;
    const int32_t * decode_codes =
        codes_for_decode.empty() ? speech_codes.data() : codes_for_decode.data();
    fprintf(stderr, "  [icl] ref_frames=%d new_frames=%d total_frames=%d prepended=%d\n",
            n_ref_frames, n_frames, total_frames, (int)!codes_for_decode.empty());
    if (const char * dp = std::getenv("QWEN3_TTS_DUMP_CODES")) {
        static std::atomic<int> dump_counter{0};
        int idx = dump_counter.fetch_add(1);
        std::string path = std::string(dp);
        if (path.find("%d") != std::string::npos) {
            char buf[1024];
            snprintf(buf, sizeof(buf), dp, idx);
            path = buf;
        }
        FILE * fp = fopen(path.c_str(), "wb");
        if (fp) {
            int32_t hdr[3] = { n_ref_frames, n_frames, n_codebooks };
            fwrite(hdr, sizeof(int32_t), 3, fp);
            if (ref_codes && n_ref_frames > 0) {
                fwrite(ref_codes, sizeof(int32_t), (size_t)n_ref_frames * n_codebooks, fp);
            }
            fwrite(speech_codes.data(), sizeof(int32_t), speech_codes.size(), fp);
            fclose(fp);
            fprintf(stderr, "  dumped ref+new codes to %s\n", path.c_str());
        }
    }

    if (!audio_decoder_.decode(decode_codes, total_frames, result.audio)) {
        result.error_msg = "Failed to decode speech codes: " + audio_decoder_.get_error();
        return result;
    }

    // (no trim needed — we decode just the generated codes, no ref prepended)
    result.t_decode_ms = get_time_ms() - t_decode_start;
    sample_memory("synth/after-decode");

    if (low_mem_mode_) {
        audio_decoder_.unload_model();
        decoder_loaded_ = false;
        sample_memory("synth/after-vocoder-unload");
    }
    
    result.sample_rate = audio_decoder_.get_config().sample_rate;
    result.success = true;
    result.t_total_ms = get_time_ms() - t_total_start;
    sample_memory("synth/end");
    
    if (params.print_timing) {
        const double audio_sec = result.sample_rate > 0
            ? (double) result.audio.size() / (double) result.sample_rate : 0.0;
        const double wall_sec = (double) result.t_total_ms / 1000.0;
        const double realtime_factor = audio_sec > 0.0 ? wall_sec / audio_sec : 0.0;
        const double x_realtime = wall_sec > 0.0 ? audio_sec / wall_sec : 0.0;
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Tokenization:    %lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Speaker encode:  %lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Code generation: %lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Vocoder decode:  %lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:           %lld ms\n", (long long)result.t_total_ms);
        fprintf(stderr, "  Audio duration:  %.2f s\n", audio_sec);
        fprintf(stderr, "  Throughput:      %.2fx realtime (RTF=%.3f)\n", x_realtime, realtime_factor);
        fprintf(stderr, "\nMemory:\n");
        fprintf(stderr, "  RSS start/end:   %s -> %s\n",
                format_bytes(result.mem_rss_start_bytes).c_str(),
                format_bytes(result.mem_rss_end_bytes).c_str());
        fprintf(stderr, "  RSS peak:        %s\n",
                format_bytes(result.mem_rss_peak_bytes).c_str());
        fprintf(stderr, "  Phys start/end:  %s -> %s\n",
                format_bytes(result.mem_phys_start_bytes).c_str(),
                format_bytes(result.mem_phys_end_bytes).c_str());
        fprintf(stderr, "  Phys peak:       %s\n",
                format_bytes(result.mem_phys_peak_bytes).c_str());
    }
    
    return result;
}

bool Qwen3TTS::ensure_encoder_loaded() {
    if (encoder_loaded_) {
        return true;
    }
    const std::string & enc_path = !speaker_encoder_model_path_.empty()
        ? speaker_encoder_model_path_
        : tts_model_path_;
    if (enc_path.empty()) {
        error_msg_ = "Internal error: missing TTS model path for lazy encoder load";
        return false;
    }
    if (!audio_encoder_.load_model(enc_path)) {
        error_msg_ = "Failed to load speaker encoder: " + audio_encoder_.get_error();
        return false;
    }
    audio_encoder_.set_abort_callback(abort_cb_, abort_data_);

    // The encoder's output (embedding_dim) is added directly into talker
    // hidden states (hidden_size floats), so the two MUST match. A silent
    // mismatch would read past the end of the embedding buffer in the talker
    // prompt assembly. Fail fast here instead.
    const int32_t enc_dim = audio_encoder_.get_config().embedding_dim;
    const int32_t hsize   = transformer_.get_config().hidden_size;
    if (enc_dim != hsize) {
        audio_encoder_.unload_model();
        error_msg_ = "Speaker encoder embedding_dim (" + std::to_string(enc_dim)
                   + ") != talker hidden_size (" + std::to_string(hsize)
                   + "); encoder GGUF does not match the TTS model";
        return false;
    }

    encoder_loaded_ = true;
    return true;
}

int32_t Qwen3TTS::get_hidden_size() const {
    return transformer_.get_config().hidden_size;
}

int32_t Qwen3TTS::get_sample_rate() const {
    return audio_decoder_.get_config().sample_rate;
}

const std::string & Qwen3TTS::get_model_type() const {
    return transformer_.get_config().model_type;
}

const std::vector<std::string> & Qwen3TTS::get_speaker_names() const {
    return transformer_.get_config().speaker_names;
}

const std::vector<int32_t> & Qwen3TTS::get_speaker_ids() const {
    return transformer_.get_config().speaker_ids;
}

bool Qwen3TTS::has_speaker_encoder() const {
    // Either: (a) the main TTS GGUF advertises speaker_encoder via metadata,
    // or (b) an external speaker_encoder GGUF was supplied via load_model_files.
    return transformer_.get_config().has_speaker_encoder
           || !speaker_encoder_model_path_.empty();
}

int32_t Qwen3TTS::get_speaker_id(const std::string & name) const {
    auto & names = transformer_.get_config().speaker_names;
    auto & ids = transformer_.get_config().speaker_ids;
    for (size_t i = 0; i < names.size(); i++) {
        if (names[i] == name) return ids[i];
    }
    return -1;
}

bool Qwen3TTS::get_speaker_embedding(const std::string & name, std::vector<float> & embedding) {
    int32_t spk_id = get_speaker_id(name);
    if (spk_id < 0) {
        error_msg_ = "unknown speaker: " + name;
        return false;
    }
    if (!transformer_.get_codec_embedding(spk_id, embedding)) {
        error_msg_ = "failed to get speaker embedding: " + transformer_.get_error();
        return false;
    }
    return true;
}

void Qwen3TTS::clear_icl_cache() {
    icl_cache_.clear();
}

void Qwen3TTS::unload_encoders() {
    if (encoder_loaded_) {
        audio_encoder_.unload_model();
        encoder_loaded_ = false;
    }
    if (codec_encoder_loaded_) {
        codec_encoder_.unload_model();
        codec_encoder_loaded_ = false;
    }
}

// -- Persistent voice caches ------------------------------------------------
//
// File layout:
//   header:
//     magic[8] = "QW3WARM\0"
//     u32 version (=1)
//     u32 model_id_len
//     model_id bytes
//     u64 prefill_cache_key      (matches at load time)
//     u64 ref_codes_hash         (matches at load time)
//   then a sequence of sections, each:
//     u32 section_id
//     u32 section_payload_size
//     section_payload bytes
//   sections are skip-tolerant; unknown ids are passed over.
//
// Sections (id):
//   1 prefill_kv_snapshot     n_pos, n_layers, layer_bytes,
//                             then K, V byte runs interleaved per layer.
//   2 icl_codec_section       n_floats, raw f32 little-endian bytes.
//   3 vocoder_icl_warmup      n_past, then four maps/vectors of float
//                             buffers (tail rings, conv_t overlaps,
//                             past_k hosts, past_v hosts).
//
// Format is intentionally simple: a malformed or quant-mismatched file
// is rejected wholesale at load time and the next synth rebuilds the
// in-memory caches from scratch.

namespace {

constexpr char     kWarmupMagic[8] = {'Q','W','3','W','A','R','M','\0'};
constexpr uint32_t kWarmupVersion  = 1;

constexpr uint32_t kSectionPrefillKv      = 1;
constexpr uint32_t kSectionIclCodec       = 2;
constexpr uint32_t kSectionVocoderWarmup  = 3;

template <typename T>
void blob_write(std::vector<uint8_t> & out, const T & v) {
    static_assert(std::is_trivially_copyable<T>::value, "POD only");
    const uint8_t * p = reinterpret_cast<const uint8_t *>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

void blob_write_bytes(std::vector<uint8_t> & out, const void * p, size_t n) {
    const uint8_t * b = static_cast<const uint8_t *>(p);
    out.insert(out.end(), b, b + n);
}

template <typename T>
bool blob_read(const std::vector<uint8_t> & in, size_t & pos, T & v) {
    if (pos + sizeof(T) > in.size()) return false;
    std::memcpy(&v, in.data() + pos, sizeof(T));
    pos += sizeof(T);
    return true;
}

bool blob_read_bytes(const std::vector<uint8_t> & in, size_t & pos,
                     void * out, size_t n) {
    if (pos + n > in.size()) return false;
    std::memcpy(out, in.data() + pos, n);
    pos += n;
    return true;
}

void serialize_float_vec(std::vector<uint8_t> & out, const std::vector<float> & v) {
    const uint32_t n = (uint32_t) v.size();
    blob_write(out, n);
    if (n) blob_write_bytes(out, v.data(), n * sizeof(float));
}

bool deserialize_float_vec(const std::vector<uint8_t> & in, size_t & pos,
                            std::vector<float> & out) {
    uint32_t n = 0;
    if (!blob_read(in, pos, n)) return false;
    out.resize(n);
    if (n) {
        if (!blob_read_bytes(in, pos, out.data(), n * sizeof(float))) return false;
    }
    return true;
}

} // namespace

bool Qwen3TTS::save_voice_warmup(const std::string & voice_id,
                                  uint64_t prefill_cache_key,
                                  uint64_t ref_codes_hash,
                                  const std::string & path,
                                  const std::string & model_id) {
    std::vector<uint8_t> blob;
    blob.reserve(8 * 1024 * 1024);

    blob_write_bytes(blob, kWarmupMagic, sizeof(kWarmupMagic));
    blob_write(blob, kWarmupVersion);
    const uint32_t mid_len = (uint32_t) model_id.size();
    blob_write(blob, mid_len);
    if (mid_len) blob_write_bytes(blob, model_id.data(), mid_len);
    blob_write(blob, prefill_cache_key);
    blob_write(blob, ref_codes_hash);

    int sections_written = 0;

    // Section 1: prefill KV snapshot.
    if (auto * snap = transformer_.get_prefill_kv_snapshot(prefill_cache_key)) {
        std::vector<uint8_t> payload;
        const int32_t n_pos     = snap->n_pos;
        const int32_t n_layers  = (int32_t) snap->k_layers.size();
        const uint32_t layer_bytes = snap->k_layers.empty()
            ? 0u
            : (uint32_t) snap->k_layers[0].size();
        blob_write(payload, n_pos);
        blob_write(payload, n_layers);
        blob_write(payload, layer_bytes);
        for (int il = 0; il < n_layers; ++il) {
            if (snap->k_layers[il].size() != layer_bytes ||
                snap->v_layers[il].size() != layer_bytes) {
                error_msg_ = "save_voice_warmup: layer byte size mismatch";
                return false;
            }
            blob_write_bytes(payload, snap->k_layers[il].data(), layer_bytes);
            blob_write_bytes(payload, snap->v_layers[il].data(), layer_bytes);
        }
        const uint32_t section_id   = kSectionPrefillKv;
        const uint32_t payload_size = (uint32_t) payload.size();
        blob_write(blob, section_id);
        blob_write(blob, payload_size);
        blob_write_bytes(blob, payload.data(), payload.size());
        ++sections_written;
    }

    // Section 2: icl_codec_section.
    if (auto * sect = transformer_.get_icl_codec_section(ref_codes_hash)) {
        std::vector<uint8_t> payload;
        serialize_float_vec(payload, *sect);
        const uint32_t section_id   = kSectionIclCodec;
        const uint32_t payload_size = (uint32_t) payload.size();
        blob_write(blob, section_id);
        blob_write(blob, payload_size);
        blob_write_bytes(blob, payload.data(), payload.size());
        ++sections_written;
    }

    // Section 3: vocoder ICL warmup state.
    auto vit = icl_cache_.find(ref_codes_hash);
    if (vit != icl_cache_.end()) {
        const auto & snap = vit->second;
        std::vector<uint8_t> payload;
        blob_write(payload, snap.n_past);
        // tail_rings (map<string, vector<float>>)
        const uint32_t n_tails = (uint32_t) snap.tail_rings.size();
        blob_write(payload, n_tails);
        for (const auto & kv : snap.tail_rings) {
            const uint32_t name_len = (uint32_t) kv.first.size();
            blob_write(payload, name_len);
            if (name_len) blob_write_bytes(payload, kv.first.data(), name_len);
            serialize_float_vec(payload, kv.second);
        }
        // conv_t_overlap_hosts
        const uint32_t n_ct = (uint32_t) snap.conv_t_overlap_hosts.size();
        blob_write(payload, n_ct);
        for (const auto & kv : snap.conv_t_overlap_hosts) {
            const uint32_t name_len = (uint32_t) kv.first.size();
            blob_write(payload, name_len);
            if (name_len) blob_write_bytes(payload, kv.first.data(), name_len);
            serialize_float_vec(payload, kv.second);
        }
        // past_k_hosts (vector<vector<float>>)
        const uint32_t n_pk = (uint32_t) snap.past_k_hosts.size();
        blob_write(payload, n_pk);
        for (const auto & v : snap.past_k_hosts) serialize_float_vec(payload, v);
        const uint32_t n_pv = (uint32_t) snap.past_v_hosts.size();
        blob_write(payload, n_pv);
        for (const auto & v : snap.past_v_hosts) serialize_float_vec(payload, v);

        const uint32_t section_id   = kSectionVocoderWarmup;
        const uint32_t payload_size = (uint32_t) payload.size();
        blob_write(blob, section_id);
        blob_write(blob, payload_size);
        blob_write_bytes(blob, payload.data(), payload.size());
        ++sections_written;
    }

    if (sections_written == 0) {
        // Nothing cached for this voice yet — don't write an empty file.
        return false;
    }

    // Atomic write via tmp + rename.
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) {
        error_msg_ = "save_voice_warmup: failed to open " + tmp;
        return false;
    }
    f.write(reinterpret_cast<const char *>(blob.data()), (std::streamsize) blob.size());
    f.close();
    if (!f) {
        error_msg_ = "save_voice_warmup: write failed for " + tmp;
        std::remove(tmp.c_str());
        return false;
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        error_msg_ = "save_voice_warmup: rename failed for " + tmp;
        std::remove(tmp.c_str());
        return false;
    }
    fprintf(stderr, "  voice-warmup: saved '%s' (%d sections, %.1f KB) → %s\n",
            voice_id.c_str(), sections_written,
            blob.size() / 1024.0, path.c_str());
    return true;
}

bool Qwen3TTS::load_voice_warmup(const std::string & path,
                                  const std::string & model_id) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        // missing file is a normal "no warmup yet" signal — don't error.
        return false;
    }
    const auto sz = f.tellg();
    if (sz <= 0) return false;
    std::vector<uint8_t> blob((size_t) sz);
    f.seekg(0);
    f.read(reinterpret_cast<char *>(blob.data()), (std::streamsize) sz);
    if (!f) {
        error_msg_ = "load_voice_warmup: short read on " + path;
        return false;
    }

    size_t pos = 0;
    char magic_in[8];
    if (!blob_read_bytes(blob, pos, magic_in, sizeof(magic_in))) return false;
    if (std::memcmp(magic_in, kWarmupMagic, sizeof(kWarmupMagic)) != 0) {
        return false;
    }
    uint32_t version = 0;
    if (!blob_read(blob, pos, version)) return false;
    if (version != kWarmupVersion) return false;

    uint32_t mid_len = 0;
    if (!blob_read(blob, pos, mid_len)) return false;
    if (mid_len > 1024) return false;
    std::string saved_model_id(mid_len, '\0');
    if (mid_len) {
        if (!blob_read_bytes(blob, pos, saved_model_id.data(), mid_len)) return false;
    }
    if (saved_model_id != model_id) {
        // quant or model swap → silently ignore
        return false;
    }

    uint64_t prefill_cache_key = 0, ref_codes_hash = 0;
    if (!blob_read(blob, pos, prefill_cache_key)) return false;
    if (!blob_read(blob, pos, ref_codes_hash))    return false;

    int sections_loaded = 0;
    while (pos < blob.size()) {
        uint32_t section_id = 0, payload_size = 0;
        if (!blob_read(blob, pos, section_id)) break;
        if (!blob_read(blob, pos, payload_size)) break;
        if (pos + payload_size > blob.size()) break;
        const size_t section_end = pos + payload_size;

        if (section_id == kSectionPrefillKv) {
            int32_t n_pos = 0, n_layers = 0;
            uint32_t layer_bytes = 0;
            if (!blob_read(blob, pos, n_pos))     { pos = section_end; continue; }
            if (!blob_read(blob, pos, n_layers))  { pos = section_end; continue; }
            if (!blob_read(blob, pos, layer_bytes)) { pos = section_end; continue; }
            if (n_pos <= 0 || n_layers <= 0 || layer_bytes == 0) {
                pos = section_end; continue;
            }
            TTSTransformer::prefill_kv_snapshot snap;
            snap.n_pos = n_pos;
            snap.k_layers.resize(n_layers);
            snap.v_layers.resize(n_layers);
            bool ok = true;
            for (int il = 0; il < n_layers && ok; ++il) {
                snap.k_layers[il].resize(layer_bytes);
                snap.v_layers[il].resize(layer_bytes);
                if (!blob_read_bytes(blob, pos, snap.k_layers[il].data(), layer_bytes)) ok = false;
                if (ok && !blob_read_bytes(blob, pos, snap.v_layers[il].data(), layer_bytes)) ok = false;
            }
            if (ok) {
                transformer_.put_prefill_kv_snapshot(prefill_cache_key, std::move(snap));
                ++sections_loaded;
            }
            pos = section_end;
        } else if (section_id == kSectionIclCodec) {
            std::vector<float> v;
            if (deserialize_float_vec(blob, pos, v)) {
                transformer_.put_icl_codec_section(ref_codes_hash, std::move(v));
                ++sections_loaded;
            }
            pos = section_end;
        } else if (section_id == kSectionVocoderWarmup) {
            AudioTokenizerDecoder::stream_state_snapshot snap;
            bool ok = blob_read(blob, pos, snap.n_past);
            uint32_t n_tails = 0, n_ct = 0, n_pk = 0, n_pv = 0;
            if (ok) ok = blob_read(blob, pos, n_tails);
            for (uint32_t i = 0; i < n_tails && ok; ++i) {
                uint32_t name_len = 0;
                if (!blob_read(blob, pos, name_len)) { ok = false; break; }
                std::string name(name_len, '\0');
                if (name_len && !blob_read_bytes(blob, pos, name.data(), name_len)) { ok = false; break; }
                std::vector<float> v;
                if (!deserialize_float_vec(blob, pos, v)) { ok = false; break; }
                snap.tail_rings.emplace(std::move(name), std::move(v));
            }
            if (ok) ok = blob_read(blob, pos, n_ct);
            for (uint32_t i = 0; i < n_ct && ok; ++i) {
                uint32_t name_len = 0;
                if (!blob_read(blob, pos, name_len)) { ok = false; break; }
                std::string name(name_len, '\0');
                if (name_len && !blob_read_bytes(blob, pos, name.data(), name_len)) { ok = false; break; }
                std::vector<float> v;
                if (!deserialize_float_vec(blob, pos, v)) { ok = false; break; }
                snap.conv_t_overlap_hosts.emplace(std::move(name), std::move(v));
            }
            if (ok) ok = blob_read(blob, pos, n_pk);
            for (uint32_t i = 0; i < n_pk && ok; ++i) {
                std::vector<float> v;
                if (!deserialize_float_vec(blob, pos, v)) { ok = false; break; }
                snap.past_k_hosts.push_back(std::move(v));
            }
            if (ok) ok = blob_read(blob, pos, n_pv);
            for (uint32_t i = 0; i < n_pv && ok; ++i) {
                std::vector<float> v;
                if (!deserialize_float_vec(blob, pos, v)) { ok = false; break; }
                snap.past_v_hosts.push_back(std::move(v));
            }
            if (ok) {
                icl_cache_.emplace(ref_codes_hash, std::move(snap));
                ++sections_loaded;
            }
            pos = section_end;
        } else {
            // unknown section — skip
            pos = section_end;
        }
    }
    if (sections_loaded == 0) return false;
    fprintf(stderr, "  voice-warmup: loaded %d sections from %s\n",
            sections_loaded, path.c_str());
    return true;
}

void Qwen3TTS::unload_model() {
    // Release in reverse-of-likely-dependency order. Each component's
    // unload_model() is idempotent (no-op if already unloaded).
    audio_decoder_.unload_model();
    codec_encoder_.unload_model();
    audio_encoder_.unload_model();
    transformer_.unload_model();

    encoder_loaded_       = false;
    codec_encoder_loaded_ = false;
    transformer_loaded_   = false;
    decoder_loaded_       = false;
    models_loaded_        = false;
}

bool Qwen3TTS::reload_model() {
    if (models_loaded_) return true;
    if (tts_model_path_.empty()) {
        error_msg_ = "reload_model(): no prior load_model_files() to reload from";
        return false;
    }
    return load_model_files(tts_model_path_, decoder_model_path_,
                            speaker_encoder_model_path_);
}

void Qwen3TTS::set_model_paths(const std::string & tts_model_path,
                               const std::string & vocoder_model_path,
                               const std::string & speaker_encoder_model_path) {
    tts_model_path_ = tts_model_path;
    speaker_encoder_model_path_ = speaker_encoder_model_path;
    if (vocoder_model_path.empty()) {
        auto slash = tts_model_path.rfind('/');
        std::string dir = (slash != std::string::npos) ? tts_model_path.substr(0, slash) : ".";
        decoder_model_path_ = dir + "/qwen3-tts-tokenizer-f16.gguf";
    } else {
        decoder_model_path_ = vocoder_model_path;
    }
}

void Qwen3TTS::set_progress_callback(tts_progress_callback_t callback) {
    progress_callback_ = callback;
}

void Qwen3TTS::set_abort_callback(ggml_abort_callback callback, void * data) {
    abort_cb_ = callback;
    abort_data_ = data;
    transformer_.set_abort_callback(callback, data);
    audio_encoder_.set_abort_callback(callback, data);
    audio_decoder_.set_abort_callback(callback, data);
}

// Mix an interleaved multi-channel buffer down to mono, storing results in `out`.
// InputT must be convertible to float; `scale` is applied before summing.
template<typename InputT>
static void mix_to_mono(const InputT * interleaved, int n_samples, int num_channels,
                        float scale, std::vector<float> & out) {
    out.resize(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < num_channels; ++c) {
            sum += static_cast<float>(interleaved[i * num_channels + c]) * scale;
        }
        out[i] = sum / num_channels;
    }
}

// Get lowercase file extension from path
static std::string get_file_extension(const std::string & path) {
    auto dot_pos = path.rfind('.');
    if (dot_pos == std::string::npos) return "";
    std::string ext = path.substr(dot_pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

// WAV file loading (16-bit PCM, 32-bit PCM, or 32-bit IEEE float)
static bool load_wav_file(const std::string & path, std::vector<float> & samples,
                          int & sample_rate) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open WAV file: %s\n", path.c_str());
        return false;
    }

    // Read RIFF header
    char riff[4];
    if (fread(riff, 1, 4, f) != 4 || strncmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "ERROR: Not a RIFF file\n");
        fclose(f);
        return false;
    }

    uint32_t file_size;
    if (fread(&file_size, 4, 1, f) != 1) {
        fclose(f);
        return false;
    }

    char wave[4];
    if (fread(wave, 1, 4, f) != 4 || strncmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "ERROR: Not a WAVE file\n");
        fclose(f);
        return false;
    }

    // Find fmt and data chunks
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sr = 0;
    uint16_t bits_per_sample = 0;

    while (!feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size;

        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            if (fread(&audio_format, 2, 1, f) != 1) break;
            if (fread(&num_channels, 2, 1, f) != 1) break;
            if (fread(&sr, 4, 1, f) != 1) break;
            fseek(f, 6, SEEK_CUR);  // Skip byte rate and block align
            if (fread(&bits_per_sample, 2, 1, f) != 1) break;

            // Handle WAVE_FORMAT_EXTENSIBLE: read actual format from SubFormat GUID
            if (audio_format == 0xFFFE && chunk_size >= 40) {
                fseek(f, 8, SEEK_CUR);  // Skip cbSize(2) + validBitsPerSample(2) + channelMask(4)
                uint16_t sub_format = 0;
                if (fread(&sub_format, 2, 1, f) != 1) break;
                audio_format = sub_format;
                // Skip remaining SubFormat GUID bytes and any extra data
                fseek(f, chunk_size - 26, SEEK_CUR);
            }
            // Skip any extra format bytes for non-extensible formats
            else if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
        }
        else if (strncmp(chunk_id, "data", 4) == 0) {
            sample_rate = sr;

            if (audio_format == 1) {  // PCM
                if (bits_per_sample == 16) {
                    int n_samples = chunk_size / (2 * num_channels);
                    std::vector<int16_t> raw(n_samples * num_channels);
                    if (fread(raw.data(), 2, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                        fclose(f);
                        return false;
                    }
                    mix_to_mono(raw.data(), n_samples, num_channels, 1.0f / 32768.0f, samples);
                }
                else if (bits_per_sample == 32) {
                    int n_samples = chunk_size / (4 * num_channels);
                    std::vector<int32_t> raw(n_samples * num_channels);
                    if (fread(raw.data(), 4, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                        fclose(f);
                        return false;
                    }
                    mix_to_mono(raw.data(), n_samples, num_channels, 1.0f / 2147483648.0f, samples);
                }
                else {
                    fprintf(stderr, "ERROR: Unsupported bits per sample: %d\n", bits_per_sample);
                    fclose(f);
                    return false;
                }
            }
            else if (audio_format == 3) {  // IEEE float
                int n_samples = chunk_size / (4 * num_channels);
                std::vector<float> raw(n_samples * num_channels);
                if (fread(raw.data(), 4, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                    fclose(f);
                    return false;
                }
                mix_to_mono(raw.data(), n_samples, num_channels, 1.0f, samples);
            }
            else {
                fprintf(stderr, "ERROR: Unsupported audio format: %d\n", audio_format);
                fclose(f);
                return false;
            }

            fclose(f);
            return true;
        }
        else {
            // Skip unknown chunk
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    fprintf(stderr, "ERROR: No data chunk found\n");
    fclose(f);
    return false;
}

// Audio file loading - dispatches to format-specific loaders based on file extension
bool load_audio_file(const std::string & path, std::vector<float> & samples,
                     int & sample_rate) {
    std::string ext = get_file_extension(path);

    if (ext == ".wav") {
        return load_wav_file(path, samples, sample_rate);
    } else {
        fprintf(stderr, "ERROR: Unsupported audio format '%s'. Supported formats: .wav\n", ext.c_str());
        return false;
    }
}

// WAV file saving (16-bit PCM at specified sample rate)
bool save_audio_file(const std::string & path, const std::vector<float> & samples,
                     int sample_rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot create WAV file: %s\n", path.c_str());
        return false;
    }
    
    // WAV header parameters
    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    uint16_t block_align = num_channels * bits_per_sample / 8;
    uint32_t data_size = samples.size() * block_align;
    uint32_t file_size = 36 + data_size;
    
    // Write RIFF header
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    
    // Write fmt chunk
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    uint16_t audio_format = 1;  // PCM
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    uint32_t sr = sample_rate;
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);
    
    // Write data chunk
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    
    // Convert float samples to 16-bit PCM and write
    for (size_t i = 0; i < samples.size(); ++i) {
        // Clamp to [-1, 1] and convert to int16
        float sample = samples[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16_t pcm_sample = (int16_t)(sample * 32767.0f);
        fwrite(&pcm_sample, 2, 1, f);
    }
    
    fclose(f);
    return true;
}

} // namespace qwen3_tts
