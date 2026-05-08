// siglip2 megakernel — stub used when GGML_CUDA=OFF so callers can
// unconditionally invoke siglip2_megakernel::install(). Returns false
// (nothing installed) and is_installed() is false; encoder dispatch falls
// through to ggml's normal CPU path.

#include "siglip2_megakernel.h"

namespace siglip2_megakernel {

bool install()       { return false; }
bool is_installed()  { return false; }

void set_active_qkv_scratch(void * /*dptr*/, std::size_t /*cap_bytes*/) {}
void clear_active_qkv_scratch() {}

bool profile_enabled() { return false; }
void profile_after_encode(const char * /*label*/, void * /*stream*/) {}
void log_device_stream_priority_range() {}

}  // namespace siglip2_megakernel
