# siglip2.cpp — Handoff

You're picking up `siglip2.cpp` after the 2026-05-07 VRAM pass + 2026-05-08 perf pass + 2026-05-09 QKV-fusion pass + 2026-05-09 megakernel-v0 Phase A0 (fused LayerNorm). Feature parity has been complete since M5; the prior passes drove VRAM from 1855 MiB to 1438 MiB and added padded MMA-FA for d_head=72; QKV fusion closed 14 % off the text path; Phase A0 ships the first custom CUDA kernel under `src/cuda/` and shaves another 3-4 % off every endpoint by collapsing 110+ LN launches per encode. Read `AGENTS.md` for the architectural overview; this doc is **what to do next + why + the user's disposition**.

## Cold-pickup orientation — if you've got a long run ahead

The user is asleep. They want to come back to "shocking news," not to a polite request for permission. The biggest swings remaining, in order of where I'd put my chips:

1. **Megakernel-style block fusion** (lever E below). Still the biggest single perf jump on the board. **The plan is now written** — see `HANDOFF-megakernel-v0.md` in this same dir. Phase A targets the text encoder at n_pos=64 (the launch-overhead-bound endpoint where the 1.43× python gap lives). ggml submodule has been fast-forwarded to pick up the graph-begin + per-op hooks needed for the dispatcher pattern; pre-existing baseline analysis + bench commands + scoping is all in that doc. Just go.

2. **Graph caching for fixed-shape encoders.** **Tried and partially blocked this pass — see "Dead ends" below.** A safe blanket-clear of `tensor->data` + `tensor->buffer` on graph_ctx tensors between calls *should* let `ggml_backend_sched_alloc_graph` re-bind compute-pool offsets cleanly, but in practice produces an illegal-memory-access on the second compute (split-graph copies in `sched->ctx` end up with offsets that collide with the cached node_alloc table). The next agent should either fork ggml's gallocr to expose a clean "rebind everything" entry point, or run with `GGML_CUDA_USE_GRAPHS` enabled *plus* graph caching simultaneously (single biggest win available — CUDA graphs need stable tensor pointers AND that's exactly what graph caching provides). Estimated 10–20 % more on text + classify_from_embeddings if landed cleanly.

3. **Padded MMA FA was already landed earlier** (`48fd16e`); see "What landed" below.

If you want a smaller warm-up before megakernel, the **banana-stand** ffn_down padding (lever J) is fully implemented — three files, ~30 LOC, measured −126 MiB VRAM at the cost of 74 µcos on one image-parity shape. **Per the user 2026-05-09: keep this parked unless you hit a 400 MiB-class win or actual OOM pressure.** 126 MiB doesn't move the needle on a 12 GB GPU and the cosine cost isn't worth it.

Stay above 0.999 cosine on the real-image path. Everything else is fair game.

## The user's standing instructions

**Priorities, all critical, no longer ranked:**
- **VRAM**: minimal, **measured against full-tower deploy mode** (both vision and text loaded — the user's actual `gather embeddings → classify` flow needs both). Currently 1504 MiB post-load / 1566 MiB post-warmup vs Python's 2322 MiB. Don't optimize for `--vision-only` at the expense of full-tower wins.
- **Performance**: as close to maximum viable for the hardware as possible. Current state vs Python at p50: /v1/embeddings **1.07×**, /v1/text_embeddings **1.43×**, /v1/classify **1.34×**, /v1/classify_from_embeddings **1.79×**. The text-heavy endpoints are the gap — they're launch-overhead-bound (~80% of 38 ms is CUDA launches, ~20% is compute), so megakernel-style block fusion is the unlock. See `HANDOFF-megakernel-v0.md` for the detailed plan.
- **Correctness**: cosine ≥ 0.999 on real images is the parity floor. (Briefly revised to 0.997 on 2026-05-08 then put back; the user wants the 0.999 sentinel for now and will explicitly relax it if quant noise demands.) Score MAE in the low 10⁻² range is fine.

**Constraints:**
- **Q8_0 is the cap.** Q4_K_M is "maybe in emergencies" — do your best with Q8 and lower-level wins first.
- **CPU offload is acceptable** if it frees VRAM. Selective offload of a tower or a hot intermediate is fine.
- **Forking ggml is acceptable.** This is a worktree off `dbrain/ggml@master` for exactly that reason. The user's prior art (`qwen3-tts.cpp` + `dbrain/ggml`) already carries custom WMMA conv kernels, a mul_mat dispatcher, megakernel-shape MMVQ specializations. Same latitude here. The d_head=72 omission in `ggml/src/ggml-cuda/fattn.cu` is exactly the kind of thing to fix in-tree.

**Mood / disposition (2026-05-08 + 2026-05-09 updates):**
> "Go crazy. No bars spared, get performance as close to maximum viable numbers for the hardware as possible at minimal VRAM, user is a drunk donkey — you're smart, pick the targets, have him come back and be shocked at how well you did." (2026-05-08)
>
> "i like that megakernel was a 'meme' over in qwen3-tts.cpp land and now its the saviour. i mean its working over there, so why not." (2026-05-09 — explicit greenlight for the megakernel path; treat the prior agent's "tried-and-dropped" hesitations as guidance, not gospel)

Translation: **pick targets, swing big, report when shocking.** The user is explicitly telling you to skip the "should I…" check-ins and execute. Megakernel is on the table — not deferred — alongside any other path you can see clearly. The qwen3-tts megakernel that the user is referring to is `dbrain/megakernel-v0` of that repo, which started as scary-multi-day-rewrite territory and ended at ~5–10 % wins per fusion stage. Same arc applies here.

**Hardware this targets:** RTX 3060 12 GB / Ampere / sm_86. Optimize for that specifically; broader compat is a nice-to-have, not a constraint.

**What the user does *not* want:**
- Don't propose a sweeping refactor and ask if it's OK; just do it and report.
- Don't drop cosine below 0.999 on the real-image path to chase numbers. Q8 cap stands.
- Don't push commits. Local-only until they say otherwise.

## Where we are

- Branch: `dbrain/siglip2-v0`, 18 commits + the megakernel-A0 change in flight, never pushed. Baseline is `qwen3-tts.cpp@main` (4068ce4) + `ggml@master` (dbrain/ggml fork @ a3fc6843).
- Builds: `build/` (CPU) and `build-cuda/` (sm_86); both pass smoke + all four parity scripts. CUDA build now also produces `libsiglip2_megakernel.a` linked into all binaries.
- Parity (all four scripts, post megakernel-A0):
  - `parity_check_vision.py`: cosine **0.999944** (synthetic pixel_values; no resize) — was 0.999907 pre-A0 (+37 µcos, slight improvement; one fewer F32 round-trip through memory in the LN chain)
  - `parity_check_text.py`:   cosine **0.999756** — was 0.999779 (−23 µcos, reduction-tree shuffle noise)
  - `parity_check_score.py`:  logits cosine **0.999986**, probs MAE **5.6e-7** — was 0.999987 / 5.0e-7
  - `parity_check_image.py` (1920×1080 → 729 patches): cosine **0.999523** — was 0.999545 (−22 µcos, same noise band)
  - **Self-parity (megakernel ON vs OFF, same C++ binary):** vision **0.999923**, text **0.999930-0.999965** across 5 prompts. Confirms the LN fusion is the only source of drift and it's well below the 0.999 floor budget.
- Live cross-check vs `kobbler-vision-1` on a 320×180 PNG, max_num_patches=729: Image embedding cosine **0.995865**, scores MAE **1.92e-2** — bit-identical to pre-fusion (Q8 noise floor, fusion doesn't move it).
- VRAM, full-tower deploy mode (both vision + text loaded, the user's actual flow):
  - Python (kobbler-vision-1):    **2322 MiB** post-warmup
  - C++ post-load (idle):         **1504 MiB** (**−818 MiB / −35 %** vs Python)
  - C++ post-warmup:              **1566 MiB** (after first `/v1/embeddings` + `/v1/text_embeddings`; +62 MiB of compute-pool activations relative to post-load)
  - For reference: `--vision-only` is **664 MiB** post-load and `--text-only` would be **~944 MiB**, but **the user's real deploy uses both towers**, so optimize for the full-tower number.
- Endpoint p50 (30 runs, both servers warmed). The 2026-05-09 measurement protocol is **one tower at a time** — kobbler-vision-1 doesn't unload itself and pollutes side-by-side numbers. The "clean GPU" column is C++ alone (kobbler-vision-1 stopped); the "shared GPU" column is the older both-running baseline kept for ratio continuity.
  - `/v1/embeddings`:               clean **55.8 ms** (was 58.2 pre-A0; **−4.1 %**); shared GPU ~58 vs Python ~54 (cpp 1.07× pre-A0 → A0 not yet re-measured against python)
  - `/v1/text_embeddings`:          clean **37.1 ms** (was 38.3 pre-A0; **−3.1 %**); shared baseline cpp 1.43× python pre-A0
  - `/v1/classify`:                 clean **90.2 ms** (was 94.1 pre-A0; **−4.1 %**); shared baseline cpp 1.34× python pre-A0
  - `/v1/classify_from_embeddings`: clean **37.1 ms** (was 38.5 pre-A0; **−3.6 %**); shared baseline cpp 1.79× python pre-A0
  - Python head-to-head re-bench is the next agent's first move post-A1 (Phase A1 = QKV-prep fusion is the bigger chunk; bench once after both land).
- Reload time: C++ cold-load **0.66 s** vs Python **10.5 s** — **~16× faster** swap (no torch/transformers import). Dominant for kobbler's vision↔TTS↔STT GPU-sharing.
- Host RAM: Python ~3 GiB → C++ ~568 MiB.

## The Siglip2Processor mask gotcha

If you touch the text path or anyone reports text drift, **remember**: HF `Siglip2Processor` for text-only returns ONLY `input_ids` (no `attention_mask`), pads to 64, and `kobbler-vision` does `model.text_model(**inputs)` straight. Pad-token-0 embeddings flow through attention as if they were real tokens. **Match this.** Passing the "correct" attention mask diverges by ~0.24 cosine on short prompts. Captured in commit `6b45933`'s body and in `siglip2_server.cpp::encode_text`'s comment.

## What landed in this pass

2026-05-07 (VRAM pass), 2026-05-08 (perf pass + housekeeping), 2026-05-09 (QKV fusion + megakernel-v0 Phase A0). In commit order:

- **`dc83baa` — chore: remove inherited TTS source** (lever H). 51 files, ~30k lines deleted. Tree now only has SigLIP2.
- **`d8bb673` — feat(convert): Q8_0 the text token embedding** (lever B). Drop `_is_keep_f16` special case for `t.token_embd.weight`. Saves ~390 MiB VRAM. GGUF on disk shrinks 1.74 → 1.47 GiB.
- **`73f3fd7` — feat(vision,text): wire `ggml_flash_attn_ext`** (lever D). Vision `build_block` + `build_probe_head` + text `build_block` all FA. K/V cast to F16, `GGML_PREC_F32` accumulator. `SIGLIP2_DISABLE_FA=1` falls back if a parity issue surfaces.
- **`b39a896` — feat(server): `--vision-only` / `--text-only`** (lever C). Flag-gates tower load; missing endpoints 503.
- **`df577f7` — perf: scheduler max_nodes 8192 → 2048** (lever G).
- **`4849c9f` — feat(preproc): antialiased bilinear CPU resize** (lever A). Separable triangular AA filter matching torchvision `F.interpolate(antialias=True)`. Lifts image cosine at heavy-downsample shapes from <0.998 to ≥0.999.
- **`48fd16e` — perf(fa): zero-pad d_head 72 → 80 to hit MMA tensor-core path.** ggml's CUDA MMA / WMMA flash-attn kernels require D % 16 == 0; SigLIP2's d_head=72 was falling back to the (CUDA-cores) tile kernel. We zero-pad Q/K/V's d-axis to next multiple of 16 before FA and slice the result back before the output projection. Zero padding is mathematically null (Q·K and V both contribute 0 in padded slots) so parity is unchanged. **+14–23 % p50 on every endpoint, same VRAM.** See the helper `fa_attn_pad_slice` in `src/siglip2_vision.cpp`.
- **`77f97f1` — docs: revert parity floor to 0.999** (housekeeping; user briefly relaxed to 0.997 on 2026-05-08 then put it back).
- **(this pass) — perf(convert,vision,text): fuse Q/K/V projections per layer.** The converter now stacks `q_proj`/`k_proj`/`v_proj` along the output dim into one `attn_qkv.{weight,bias}` per encoder block, and `build_block` runs a single wider mul_mat + bias-add instead of three. Q/K/V become *strided* `ggml_view_3d` views into the fused output (no `ggml_reshape_3d` step — non-contiguous along positions), and the existing `ggml_cont`-before-`ggml_pad` path materialises them when the d_head-pad helper needs contiguous memory, so parity is unchanged. **−6 ms (−14 %) on text and classify_from_embeddings, −7 ms (−7 %) on classify, −1 ms (−2 %) on embeddings.** Image-parity cosine on the heavy-downsample shape *improved* from 0.999054 → 0.999545 (single matmul has slightly better fp32 accumulator behaviour than three independent ones). 81 fewer kernel launches per request × both towers loaded.

- **(this pass) — perf(cuda): megakernel-v0 Phase A0 — fused LayerNorm-with-affine.** First custom CUDA kernel under `src/cuda/`. ggml-cuda exposes graph-begin and per-op hooks (already plumbed in the ggml fork at `a3fc6843`); `siglip2_megakernel::install()` wires `siglip2_graph_begin_hook` (scans every cgraph for the `ggml_norm → ggml_mul-w → ggml_add-b` chain shape, anchors the plan on the ADD's dst, marks norm + mul as followers) and `siglip2_op_hook` (fires one `fused_layernorm_affine_kernel` per anchor, returns true for followers). Same one-pass mean+var formula as `ggml/src/ggml-cuda/norm.cu`; the affine multiplies use `__fmul_rn` / `__fadd_rn` to suppress FMA fusion so the result matches ggml's split (norm → mul → add) per-store-rounding bit-for-bit modulo the reduction-tree shuffle. Plan-builder records **55 LN groups for text** (27 × 2 + final_ln) and **56 for vision** (27 × 2 + probe_head + post_ln), so each text encode drops 110 launches and each vision encode drops 112; `/v1/classify` (vision + 5 texts) drops 662. Wired into siglip2-server, siglip2-cli, siglip2-text-cli; CPU-only builds (`GGML_CUDA=OFF`) get a stub `install()` that returns false. Kill switch: `SIGLIP2_DISABLE_MEGAKERNEL=1`. Verbose graph-begin counts: `SIGLIP2_MEGAKERNEL_VERBOSE=1`. **Clean-GPU A/B (kobbler-vision-1 stopped):**
  - `/v1/embeddings`:               58.2 → 55.8 ms (**−4.1 %**)
  - `/v1/text_embeddings`:          38.3 → 37.1 ms (**−3.1 %**)
  - `/v1/classify`:                 94.1 → 90.2 ms (**−4.1 %**)
  - `/v1/classify_from_embeddings`: 38.5 → 37.1 ms (**−3.6 %**)

  Self-parity (megakernel ON vs OFF, same C++ binary): vision cosine **0.99992**, text **0.99993-0.99997** across 5 prompts. HF parity (post-A0): vision **0.999944**, text **0.999756**, image-1920×1080→729 **0.999523**, score logits **0.999986** / probs MAE 5.6e-7 — all comfortably above the 0.999 floor.

  3-4 % is small for a custom-kernel commit, but it validates the scaffolding (hooks land, plan-builder finds the right node count, kill-switch works, parity holds) and is the foundation for the bigger fusions ahead. Per-launch cost on these small ops (norm, mul) is ~10-12 µs, not the 12-15 µs estimated; the bigger wins live in the **QKV-prep chain** (3 cont + 3 pad + 2 cast = 8 launches/block, fuseable to one custom split-permute-pad-cast kernel — Phase A1, see `HANDOFF-megakernel-v0.md`).

## Dead ends and partial attempts (2026-05-09 pass)

These are flagged so the next agent doesn't redo them or wastes the same context window.

### Graph caching attempt — both shared-sched and per-entry-sched paths blocked

Tried to cache `(graph_ctx, gf, input/output tensor pointers)` per shape and reuse across calls so we'd skip the per-call build_block traversal *and* enable CUDA-graph reuse (which needs stable tensor pointers to warm up).

**Variant 1 (shared scheduler):** clear `data`/`buffer` on every cached compute-pool tensor before each `ggml_backend_sched_alloc_graph`. First call works; second call crashes with `illegal memory access` on the first compute kernel (varies — GELU, MUL, etc.).

**Variant 2 (per-cache-entry scheduler):** each cache entry holds its own `ggml_backend_sched_t`, so the gallocr's `node_allocs[]` table is exclusive to that entry. **Same crash.** The per-sched isolation does NOT fix the bug, which means the issue is *not* multi-shape interference — it's something in how gallocr re-uses its offset table across calls on the same graph.

Diagnosis (deeper): `ggml_backend_sched_split_graph` frees `sched->ctx` and creates fresh split-copy tensors at the same context-buffer offsets each call. Even when their gallocr is per-entry, the internal `sched->graph.nodes[]`/`leafs[]` tables may include split-copy tensors whose pointers cycle each call. `ggml_gallocr_needs_realloc` returns false because the leaf/node *counts* match across calls (graph is identical), so the path skips the full-reserve fallback that would re-derive offsets. But the saved `node_allocs[]` is keyed by position in `sched->graph` — and on the second call, position N may point to a *fresh* split-copy tensor whose `data == NULL`, triggering an allocation at the saved offset that overlaps with one of our cached tensors that *also* has `data` cleared and gets allocated at the same address.

Bottom line: Both variants land Variant-1's crash. The fix isn't "isolate gallocrs" — it's "force the gallocr to re-derive offsets fresh each call when our tensor data is cleared." Practical paths the next agent should explore:

1. **Fork ggml's gallocr** to add an explicit "re-reserve and re-bind" entry point that doesn't skip on `data != NULL`. Probably ~50 LOC in `src/ggml-alloc.c`. The cleanest unlock.
2. **Skip the scheduler entirely.** Use raw `ggml_backend_graph_compute` with a fixed single-backend path (drop the CPU fallback for vision/text encoder; verify all ops have CUDA kernels — they do). Manually manage one pinned compute buffer with offsets baked in once. More work but no scheduler-internal state to fight.
3. **Cache the build but not the gf.** Keep `graph_ctx` alive across calls (skipping build_block — the heavy step), call `ggml_graph_clear` and `ggml_build_forward_expand` again from the cached output tensor each call, then alloc/compute/reset as before. Saves the construction cost without reusing gallocr state. Should not crash.

Path 3 is the lowest-risk experiment to start with. (I didn't try it this pass — context budget.)

### CUDA graphs (`-DGGML_CUDA_GRAPHS=ON`) — built, measured, no-op (slight regression)

Rebuilt with the flag on, expecting the warmed-up CUDA-graph path to kick in after two calls. It doesn't, because `ggml_cuda_graph_get_key(cgraph) = cgraph->nodes[0]` and our first node is a freshly-allocated tensor pointer on every call. Even if the key matched, `ggml_cuda_graph_update_required` does a `memcmp(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor))` which always differs because the tensors themselves are fresh objects. Net effect with the flag on: small overhead from the compatibility/warmup checks, no benefit (text path crept up ~10 ms). **Reverted to OFF.** This re-enables only after graph caching above is solved.

### Lever J (ffn_down padding) — confirmed parked

Per the user 2026-05-09: 126 MiB VRAM win is not worth a sub-0.999 cosine drop. The bar to drop below 0.999 is a 400 MiB-class win or actual OOM pressure (neither applies on a 12 GB GPU with vision-only at 720 MiB resident). Implementation details are still in lever J below; left documented in case the deploy substrate changes.

## Lever queue — what's left

Ordered for an overnight run: biggest swings first, smaller-but-clean wins below, and the "documented-but-don't-land-without-asking" entries last.

### E. Megakernel-style block fusion *(perf + VRAM; biggest swing)*
- For SigLIP2 vision, an entire pre-LN transformer block (attn + MLP + 2 residuals) fuses into one CUDA kernel at the dominant shape (n_pos=729, H=1152, d_head=72). Eliminates intermediate-tensor allocations entirely, drives kernel launches per layer from ~10 to 1, and lets you pick the optimal MMA layout for these specific shapes (where ggml's generic dispatcher leaves perf on the table even after the d_head=72 → 80 padding hack).
- **Reference:** the user has prior art at `kobbler/docker/tts-qwen3-dev/HANDOFF-megakernel-v0.md` (the qwen3-tts megakernel-v0 plan, branch `dbrain/megakernel-v0` of that repo). Phase A is "shape-specialized MMVQ + per-layer fused kernels at one fixed shape," Phase B widens it. The pattern translates directly to SigLIP2's vision blocks at production max_num_patches=729.
- **Why this is the right next move:** with padded MMA-FA already in (lever D + `48fd16e`), the next biggest perf lift comes from killing intermediate allocations and kernel launches in the residual+LN+MLP chain — exactly what a fused per-block kernel buys.
- **How to start:** read the qwen3-tts handoff doc, then sketch a `HANDOFF-megakernel-v0.md` here mirroring its structure. Phase A target = parity-clean fused vision encoder block at fixed n_pos=729, H=1152, d_head=72. Q8 weights, F32 activations, FA path inside the kernel (no separate FA dispatch).
- **Risk:** Real. Massive-rewrite-of-internal-GGML territory. User explicitly opted in. Expect 1–2 weeks if you go all the way through Phase A + B; less if you stop at a single-shape Phase A drop-in.

### Graph caching for fixed-shape encoders *(perf; medium — bumped to "do this BEFORE megakernel, it unblocks CUDA graphs")*
- Right now both `VisionEncoder::encode()` and `TextEncoder::encode()` allocate the per-call arena, build the cgraph from scratch, and run `ggml_backend_sched_alloc_graph` every call. For text n_pos is constant (=64) so this is pure overhead repeated forever; for vision the shape varies but is bounded — there's a finite set of `(n_patches_h, n_patches_w)` shapes the binary-search lands on for any fixed `max_num_patches`.
- **Suspected payoff:** Even after QKV fusion, text is at 1.43× python (38 ms vs 26 ms). With graph caching + CUDA graphs (which need stable tensor pointers and depend on caching first) this should be ≤1.0×.
- **What was tried this pass (failed):** see "Dead ends" above. The shared-scheduler path runs into gallocr offset reuse issues across calls.
- **Recommended path:** one `ggml_backend_sched_t` per cache entry. Pay the small extra activation-buffer VRAM (call it ~50–100 MiB per cached shape; vision-only deploy still well under 1 GiB). Single-shape text cache + a 2–3 entry vision LRU. Keyed on `(n_pos)` for text, `(n_patches_h, n_patches_w)` for vision. Then enable `-DGGML_CUDA_GRAPHS=ON` and the warmup path takes over after two stable calls.
- **Risk:** Medium. The per-shape VRAM cost is the main concern; the user has explicitly OK'd "go crazy" on perf, and ~100 MiB extra activation overhead beats the 10–20 % perf left on the table.

### Endpoint perf — bonus *(small; complements megakernel/graph-cache)*
- **Pinned-memory upload paths.** `ggml_backend_tensor_set` with pageable host memory is a known small loss. Pinned alloc for the I/O scratch on the encoder side could shave a few ms off small-batch endpoints (text + classify_from_embeddings).
- **`return_last_hidden` /v1/embeddings flag** (501-stubbed today). Build a separate vision encode-with-hidden path. No user has asked but it'd unblock anyone wanting per-patch features.

### F. Selective CPU offload *(VRAM lever; modest, situational)*
- The text token embedding (~295 MiB Q8) is now the largest single tensor. Moving it to CPU saves that much VRAM at the cost of a host→device transfer of `n_tokens` rows (≤64) per text request — negligible bandwidth. Only worth pulling for a vision-heavy deploy where text fires occasionally.
- **How:** `ggml_backend_sched_set_tensor_backend` for `t.token_embd` after weight allocation, pin `ggml_get_rows` to CPU, let the scheduler split.
- **Risk:** Per-call bandwidth tax for tiny fetches. Should be invisible.

### J. ffn_down Q8_0 inner-axis padding *(VRAM lever; "always money in the banana stand")*
- **Status:** Implemented and validated 2026-05-08, then deliberately not landed. Documented here so the user can pull it when VRAM next gets tight (kobbler's vision↔TTS↔STT GPU sharing only gets harder over time).
- ffn_down's innermost dim is **4304** — not divisible by Q8_0's block size 32 — so 27 vision + 27 text + 1 probe-head ffn_downs all fall back to F16. Pad the inner axis to 4320 (next multiple of 32) at conversion time, mirror the pad with a graph-side `ggml_pad` on the GELU output before `ffn_down`. Zeros × Q8 weights = zeros so the padded slots contribute nothing to the result.
- **Measured payoff (full-tower server, RTX 3060):**
  - VRAM:   1438 MiB → 1312 MiB  (**−126 MiB / −9 %**)
  - Disk:   1.47 GiB → 1.21 GiB
  - Quality: image cosine on 1920×1080 → 729 patches drops **0.999042 → 0.998974** (74 µcos below the 0.999 floor). 512×512→729 *improves* to 0.999754. Vision/text/score parity all unchanged at ≥ 0.99972. Functionally invisible for ranking.
- **Why it's documented, not landed:** user wants the 0.999 floor as a sentinel for now. **Don't land this without explicit OK** — it's a "we'll talk if it drops" item per their 2026-05-08 message. If you bring it up: have the data ready and let them call it.
- **How to land (when greenlit):** three files, ~30 LOC total.
  - `scripts/convert_siglip2_to_gguf.py::_convert_dtype`, in the `q8_0` branch before `gguf.quants.quantize`: detect `ggml_name.endswith("ffn_down.weight")` and `np.pad` the last numpy axis to next multiple of 32.
  - `src/siglip2_vision.cpp::build_block` and `build_probe_head`, before `ggml_mul_mat(ctx, layer.down_w, cur)`: `if (layer.down_w->ne[0] > cur->ne[0]) cur = ggml_pad(ctx, cur, layer.down_w->ne[0] - cur->ne[0], 0, 0, 0);`.
  - Same one-liner in `src/siglip2_text.cpp::build_block`.
  - Re-convert, parity, bench, done.

## Build / test cheat sheet

```bash
# CPU build
docker run --rm -v $PWD:/src -v siglip2-cpp-ccache:/root/.ccache -w /src \
  tts-qwen3-dev:builder bash -lc \
  'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF && cmake --build build -j'

# CUDA build (NEEDS --gpus all for sm_86 detection; otherwise nvcc warns + falls back)
docker run --rm --gpus all -v $PWD:/src -v siglip2-cpp-ccache:/root/.ccache -w /src \
  tts-qwen3-dev:builder bash -lc \
  'cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build-cuda -j'

# Convert HF -> GGUF (run via kobbler-vision image which has torch+transformers)
docker run --rm -v $PWD:/work -w /work kobbler-vision bash -c \
  'pip install --quiet --break-system-packages gguf 2>/dev/null
   python3 scripts/convert_siglip2_to_gguf.py \
     --input  reference/hf/siglip2-so400m-patch16-naflex \
     --output models/siglip2-so400m-naflex-q8_0.gguf \
     --type   q8_0'

# Parity tests (any kobbler-vision-image-shaped container with libgomp1):
#   parity_check_vision.py  — vision encoder vs HF Siglip2VisionModel
#   parity_check_text.py    — text encoder vs HF Siglip2TextModel
#   parity_check_score.py   — sigmoid scoring vs HF Siglip2Model
#   parity_check_tokenizer.py — sentencepiece vs HF AutoTokenizer
#   parity_check_image.py   — end-to-end image -> embedding parity
# All print "PASS" / "FAIL" with a concrete cosine + threshold.

# Live bench vs running kobbler-vision-1 (lite = VRAM + quality only)
docker run -d --rm --name siglip2-server-cuda --gpus all --network host \
  -v $PWD:/work:ro -w /work tts-qwen3-dev:builder \
  bash -lc '/work/build-cuda/siglip2-server \
    --model /work/models/siglip2-so400m-naflex-q8_0.gguf \
    --tokenizer /work/reference/hf/siglip2-so400m-patch16-naflex/tokenizer.model \
    --port 18890'
docker run --rm --gpus all --network host -v $PWD:/work -w /work kobbler-vision bash -c \
  'pip install --quiet --break-system-packages requests 2>/dev/null
   python3 scripts/bench_vs_python.py \
     --python-url http://localhost:8890 \
     --cpp-url    http://localhost:18890 \
     --image-path models/test_image.png \
     --lite --max-num-patches 729'
```

## Working assumptions

- The submodule worktrees (`siglip2.cpp` and `siglip2.cpp/ggml`) are dedicated — modify ggml freely. The user has prior form forking ggml for a related project (`dbrain/ggml@master`); you have the same latitude here.
- Commits at logical milestones; never push without explicit OK.
- One feat/fix per logical change, descriptive commit bodies. User reviews via `git log`.
- If you ship a megakernel-class change, write a `HANDOFF-<name>.md` documenting it the way the user did for `qwen3-tts.cpp` (see `kobbler/docker/tts-qwen3-dev/HANDOFF-megakernel-v0.md`). Future-you will thank you.
- **Bench discipline.** Live perf numbers wobble because the GPU is shared with a running `kobbler-vision-1`, `qwen3-tts-server`, etc. Quote ratios over absolute ms; warm both servers fully before measuring; report run-to-run variance if it matters.

## What "done" looks like

User's bar: "ridiculously optimized on this hardware." Concretely:
- ✅ Full-tower VRAM well below Python (now **1504 MiB** post-load vs Python **2322 MiB** — −35 %).
- ✅ Cosine ≥ 0.999 on real images (1920×1080 → 729 patches: **0.999545** post-QKV-fusion).
- ✅ No quality regression from Q8 + FA + padded-MMA + QKV-fusion (all parity scripts pass at the 0.999 floor).
- 🟡 Performance: padded MMA-FA + QKV fusion have closed the easier slices of the gap. Remaining gap is **CUDA launch-overhead on text-heavy endpoints** (~80 % of text encode time is launches, not compute). Megakernel is the unblock — see `HANDOFF-megakernel-v0.md`. Land it and text should drop to ≤1× Python.
- ✅ Cold-load: ~16× faster than Python.

Ship a commit + update this doc's "Where we are" section as you land things. The next agent (or future-you) will pick up from there.

Now go.
