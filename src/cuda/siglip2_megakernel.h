// siglip2 megakernel — install API.
//
// Pure C++ header (no CUDA dependencies), safe to include from any TU. The
// kernels themselves live in siglip2_megakernel.cuh / .cu and are linked into
// the siglip2_megakernel static lib (built only when GGML_CUDA=ON).
//
// Phase A0 fuses the 3-op LayerNorm-with-affine chain (norm + mul-with-weight
// + add-with-bias) into a single CUDA kernel. The plan-builder anchors on the
// add node (every text + vision LN site has both weight and bias), so the
// op-hook can dispatch the fused kernel writing directly into the add's dst,
// and the upstream norm and mul nodes short-circuit as followers. ~164 launch
// reductions per text encode, ~166 per vision encode, ~786 per /v1/classify.
//
// Bigger fusions (per-block megakernel, fused mul_mat+bias) come in later
// phases.

#pragma once

#include <cstddef>

namespace siglip2_megakernel {

// Install the per-op + graph-begin hooks in ggml-cuda. Idempotent — call once
// during process init (before any encoder->load() so the hooks are live for
// the first graph compute). Returns true if the hooks were registered, false
// if already installed or if SIGLIP2_DISABLE_MEGAKERNEL=1 in the env.
bool install();

// Did install() succeed in this process? Useful for boot logging.
bool is_installed();

// QKV-prep scratch is owned by the encoder (one slab per encoder, sized
// for the encoder's worst-case shape). The encoder calls set_active_qkv_scratch
// on the host thread that's about to call ggml_backend_graph_compute, then
// (optionally) clear_active_qkv_scratch when done. The pointer is stored in
// thread_local state so concurrent encodes (vision + text on private streams
// from /v1/classify) don't share a slab. Pass `cap_bytes == 0` or
// `dptr == nullptr` to disable QKV-prep fusion for this encode (the megakernel
// hook will then return false and fall back to ggml's split path). Pointer
// must remain valid until the matching graph_compute returns; pre-allocating
// at load() and freeing at encoder destruction is the recommended pattern,
// because captured CUDA graphs bake the pointer into recorded launches.
void set_active_qkv_scratch(void * dptr, std::size_t cap_bytes);
void clear_active_qkv_scratch();

// ─── op profiling (env-gated, low-overhead per-anchor CUDA events) ───────────
//
// SIGLIP2_PROFILE_OPS=1 enables per-anchor cudaEventRecord at every megakernel
// hook fire, captured into the CUDA graph alongside the actual launches. After
// graph_compute returns, the encoder calls profile_after_encode() to sync the
// last event, walk the recorded ranges, and log an aggregate breakdown by
// anchor type ("LN", "QKV-copy", "QKV-split", "post-FA cont", "tail-bias+res",
// "tail-bias+gelu") plus the gaps between anchors (= ggml's mul_mat / FA /
// other unfused ops). Designed so it can be enabled at boot and left on with
// minimal overhead — events are pooled and reused across replays.
//
// Must be set BEFORE the first graph_compute (the events get captured into
// the graph at warmup-2; turning on later won't retroactively instrument the
// already-captured graph).
//
// Tag values must match those passed by the megakernel internally. Encoders
// don't need to know them — they're internal to the instrumentation.
bool profile_enabled();   // returns the cached env-flag value

// Encoder-side calls. label distinguishes "vision" vs "text" in the dumped
// breakdown. Safe no-op when profile is disabled. Must be invoked on the
// same thread that ran graph_compute. The stream param is currently unused
// (kept for forward compat); pass nullptr.
void profile_after_encode(const char * label, void * stream);

// Logs the device's cudaDeviceGetStreamPriorityRange to stderr. Useful as a
// boot-time diagnostic to confirm whether HIGH/LOW priorities can actually
// take effect (consumer Ampere is [-5,0] so LOW is a no-op there). Safe to
// call from CPU-only builds (the stub returns without printing).
void log_device_stream_priority_range();

}  // namespace siglip2_megakernel
