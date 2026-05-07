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
// phases — see HANDOFF-megakernel-v0.md.

#pragma once

namespace siglip2_megakernel {

// Install the per-op + graph-begin hooks in ggml-cuda. Idempotent — call once
// during process init (before any encoder->load() so the hooks are live for
// the first graph compute). Returns true if the hooks were registered, false
// if already installed or if SIGLIP2_DISABLE_MEGAKERNEL=1 in the env.
bool install();

// Did install() succeed in this process? Useful for boot logging.
bool is_installed();

}  // namespace siglip2_megakernel
