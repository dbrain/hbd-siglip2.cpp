// siglip2 megakernel — stub used when GGML_CUDA=OFF so callers can
// unconditionally invoke siglip2_megakernel::install(). Returns false
// (nothing installed) and is_installed() is false; encoder dispatch falls
// through to ggml's normal CPU path.

#include "siglip2_megakernel.h"

namespace siglip2_megakernel {

bool install()       { return false; }
bool is_installed()  { return false; }

}  // namespace siglip2_megakernel
