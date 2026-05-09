#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>
#include <memory>

namespace siglip2 {

// Generic GGUF model loader class
// This is a simplified loader that can be extended for specific model types
class GGUFLoader {
public:
    GGUFLoader();
    ~GGUFLoader();
    
    // Open GGUF file and parse metadata
    bool open(const std::string & path);
    
    // Close file and free resources
    void close();
    
    // Get error message if operation failed
    const std::string & get_error() const { return error_msg_; }
    
    // Get number of tensors in file
    int64_t get_n_tensors() const;
    
    // Get tensor name by index
    const char * get_tensor_name(int64_t idx) const;
    
    // Get tensor type by index
    enum ggml_type get_tensor_type(int64_t idx) const;
    
    // Get tensor offset by index
    size_t get_tensor_offset(int64_t idx) const;
    
    // Get tensor size by index
    size_t get_tensor_size(int64_t idx) const;
    
    // Get metadata value (returns -1 if not found)
    int32_t get_u32(const char * key, int32_t default_val = 0) const;
    float get_f32(const char * key, float default_val = 0.0f) const;
    
    // Get data offset (start of tensor data in file)
    size_t get_data_offset() const;
    
    // Get GGUF context (for advanced usage)
    struct gguf_context * get_ctx() const { return ctx_; }
    
    // Get metadata context
    struct ggml_context * get_meta_ctx() const { return meta_ctx_; }
    
protected:
    struct gguf_context * ctx_ = nullptr;
    struct ggml_context * meta_ctx_ = nullptr;
    std::string error_msg_;
    std::string file_path_;
};

// Helper function to allocate and load tensor data from GGUF file.
// preferred_backend_type is required (no default): an implicit CPU default
// would silently route weight buffers through CPU on machines without the
// requested device, dragging the encode graph onto CPU. Callers pick
// explicitly so the CPU-vs-GPU intent is visible at the call site.
bool load_tensor_data_from_file(
    const std::string & path,
    struct gguf_context * ctx,
    struct ggml_context * model_ctx,
    const std::map<std::string, struct ggml_tensor *> & tensors,
    ggml_backend_buffer_t & buffer,
    std::string & error_msg,
    enum ggml_backend_dev_type preferred_backend_type
);

// Helper to initialize backend with GPU preference and CPU fallback
ggml_backend_t init_preferred_backend(const char * component_name, std::string * error_msg);
void release_preferred_backend(ggml_backend_t backend);

// Like init_preferred_backend but bypasses the process-wide singleton — every
// call returns a fresh backend with its own CUDA stream(s). Intended for
// concurrent encoders that want overlapping GPU execution (siglip2 server's
// /v1/classify runs vision + text on private streams). Free with
// release_preferred_backend (already handles the non-shared case).
//
// stream_priority hint matches ggml_cuda_stream_priority:
//    0 = DEFAULT, 1 = LOW, -1 = HIGH. Non-CUDA backends ignore it. On
// devices with a single priority level (consumer Ampere has [-5,0], so LOW
// is a no-op there; HIGH wins) the hint is best-effort.
ggml_backend_t init_separate_backend(const char * component_name, std::string * error_msg,
                                     int stream_priority = 0);

// Helper function to free model resources
void free_ggml_resources(struct ggml_context * ctx, ggml_backend_buffer_t buffer);

} // namespace siglip2
