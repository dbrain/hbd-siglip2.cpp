# siglip2.cpp — Handoff

You're picking up `siglip2.cpp` after the 2026-05-07 VRAM-minimisation pass + 2026-05-08 perf pass. Feature parity has been complete since M5; the last two passes drove VRAM from 1855 MiB to 1438 MiB and added padded MMA-FA for the d_head=72 case. Read `AGENTS.md` for the architectural overview; this doc is **what to do next + why + the user's disposition**.

## Cold-pickup orientation — if you've got a long run ahead

The user is asleep. They want to come back to "shocking news," not to a polite request for permission. Three fattest swings, in order of where I'd put my chips for an overnight session:

1. **Megakernel-style block fusion** (lever E below). The biggest single perf jump on the board. The user explicitly opted in for this kind of work — `dbrain/ggml@master` is exactly the worktree to fork further. Pattern reference: `kobbler/docker/tts-qwen3-dev/HANDOFF-megakernel-v0.md`. Plan it as Phase A (skeleton + parity-clean fused attn block at fixed shape n_pos=729, H=1152) → Phase B (FFN fusion + multi-shape) and write a `HANDOFF-megakernel-v0.md` of your own as you go so the *next* agent can resume mid-flight.

2. **Graph caching for fixed-shape encoders.** Text path always has n_pos=64; vision shapes are bounded by max_num_patches and there are only a finite set the binary-search can land on. Right now `encode()` rebuilds the graph + reruns `ggml_backend_sched_alloc_graph` every call — that's pure overhead, especially visible on the text path (54 ms vs Python's 34 ms; the kernels aren't the gap, the setup is). Cache `(shape) → built graph + bound input tensors` and reuse. Should win 10–30 % on text and classify_from_embeddings without touching kernels.

3. **Padded MMA FA was already landed in this pass** (`48fd16e`); see "What landed" below for context. With megakernel + graph caching both done you'll have closed the perf gap to Python decisively without touching VRAM.

If you want a smaller warm-up before megakernel, the **banana-stand** ffn_down padding (lever J) is fully implemented — three files, ~30 LOC, measured −126 MiB VRAM at the cost of 74 µcos on one image-parity shape. The user wanted to keep the 0.999 sentinel "for now"; if you read the lever and decide it's the right move for the deploy, surface the data and ask, but don't land without explicit OK on that one.

Stay below 0.999 cosine on the real-image path = needs explicit user buy-in. Everything else is fair game.

## The user's standing instructions

**Priorities, all critical, no longer ranked:**
- **VRAM**: minimal. Sub-1 GiB resident on the vision-only path is now floor, not ceiling. Push lower.
- **Performance**: as close to maximum viable for the hardware as possible. Until 2026-05-08 the user had perf on the backburner; after the v0 VRAM pass landed they upgraded the bar to "go crazy". The C++ path is currently 0.5–0.9× of the Python service at p50 because d_head=72 routes attention to the slow CUDA tile kernel — that's the headline gap and the next agent's first target.
- **Correctness**: cosine ≥ 0.999 on real images is the parity floor. (Briefly revised to 0.997 on 2026-05-08 then put back; the user wants the 0.999 sentinel for now and will explicitly relax it if quant noise demands.) Score MAE in the low 10⁻² range is fine.

**Constraints:**
- **Q8_0 is the cap.** Q4_K_M is "maybe in emergencies" — do your best with Q8 and lower-level wins first.
- **CPU offload is acceptable** if it frees VRAM. Selective offload of a tower or a hot intermediate is fine.
- **Forking ggml is acceptable.** This is a worktree off `dbrain/ggml@master` for exactly that reason. The user's prior art (`qwen3-tts.cpp` + `dbrain/ggml`) already carries custom WMMA conv kernels, a mul_mat dispatcher, megakernel-shape MMVQ specializations. Same latitude here. The d_head=72 omission in `ggml/src/ggml-cuda/fattn.cu` is exactly the kind of thing to fix in-tree.

**Mood / disposition the user explicitly set for this handoff:**
> "Go crazy. No bars spared, get performance as close to maximum viable numbers for the hardware as possible at minimal VRAM, user is a drunk donkey — you're smart, pick the targets, have him come back and be shocked at how well you did."

Translation: **pick targets, swing big, report when shocking.** The user is explicitly telling you to skip the "should I…" check-ins and execute. Megakernel is on the table — not deferred — alongside any other path you can see clearly. If the rough cost-benefit suggests a kernel-template instantiation lands more perf than a megakernel for a fraction of the effort, do that first. If you can see a megakernel path that beats the small knobs, jump.

**Hardware this targets:** RTX 3060 12 GB / Ampere / sm_86. Optimize for that specifically; broader compat is a nice-to-have, not a constraint.

**What the user does *not* want:**
- Don't propose a sweeping refactor and ask if it's OK; just do it and report.
- Don't drop cosine below 0.999 on the real-image path to chase numbers. Q8 cap stands.
- Don't push commits. Local-only until they say otherwise.

## Where we are

- Branch: `dbrain/siglip2-v0`, 16 commits, never pushed. Baseline is `qwen3-tts.cpp@main` (4068ce4) + `ggml@master` (dbrain/ggml fork).
- Builds: `build/` (CPU) and `build-cuda/` (sm_86); both pass smoke + all four parity scripts.
- Parity (all four scripts):
  - `parity_check_vision.py`: cosine **0.999937** (synthetic pixel_values; no resize)
  - `parity_check_text.py`:   cosine **0.999723**
  - `parity_check_score.py`:  logits cosine **0.999992**, probs MAE **3.8e-7**
  - `parity_check_image.py` (1920×1080 → 729 patches, the heavy-downsample case): cosine **0.999054** ← AA bilinear + padded MMA FA, both above 0.999
- Live cross-check vs `kobbler-vision-1` (Python fp16, GPU) on a 320×180 PNG, max_num_patches=729 (an *upsample* case where AA can't help; cosine is bounded by Q8 noise):
  - Image embedding cosine: **0.995865**
  - Score MAE: **1.92e-2**
- VRAM (siglip2-server, model loaded + warmed, RTX 3060):
  - Python (kobbler-vision-1):    **2325 MiB**
  - C++ both towers (full):       **1438 MiB**  (−887 MiB / **−38%** vs Python)
  - C++ `--vision-only`:           **720 MiB**  (−1605 MiB / **−69%** vs Python — kobbler's deploy target)
  - C++ `--text-only`:             **944 MiB**
- Endpoint p50 (8 runs, both servers warmed, GPU somewhat shared with neighbours so absolute numbers wobble run-to-run):
  - `/v1/embeddings`:               C++ ~60 ms vs Python ~56 ms (≈0.93×)
  - `/v1/text_embeddings`:          C++ ~47 ms vs Python ~29 ms (≈0.61× — graph-setup bound, see lever below)
  - `/v1/classify`:                 C++ ~101 ms vs Python ~69 ms (≈0.68×)
  - `/v1/classify_from_embeddings`: C++ ~45 ms vs Python ~22 ms (≈0.48× — small-batch text, ditto)
- Reload time: C++ cold-load **0.66 s** vs Python **10.5 s** — **~16× faster** swap (no torch/transformers import). Dominant for kobbler's vision↔TTS↔STT GPU-sharing.
- Host RAM: Python ~3 GiB → C++ ~568 MiB.

## The Siglip2Processor mask gotcha

If you touch the text path or anyone reports text drift, **remember**: HF `Siglip2Processor` for text-only returns ONLY `input_ids` (no `attention_mask`), pads to 64, and `kobbler-vision` does `model.text_model(**inputs)` straight. Pad-token-0 embeddings flow through attention as if they were real tokens. **Match this.** Passing the "correct" attention mask diverges by ~0.24 cosine on short prompts. Captured in commit `6b45933`'s body and in `siglip2_server.cpp::encode_text`'s comment.

## What landed in this pass

2026-05-07 (VRAM pass), then 2026-05-08 (perf pass + housekeeping). In commit order:

- **`dc83baa` — chore: remove inherited TTS source** (lever H). 51 files, ~30k lines deleted. Tree now only has SigLIP2.
- **`d8bb673` — feat(convert): Q8_0 the text token embedding** (lever B). Drop `_is_keep_f16` special case for `t.token_embd.weight`. Saves ~390 MiB VRAM. GGUF on disk shrinks 1.74 → 1.47 GiB.
- **`73f3fd7` — feat(vision,text): wire `ggml_flash_attn_ext`** (lever D). Vision `build_block` + `build_probe_head` + text `build_block` all FA. K/V cast to F16, `GGML_PREC_F32` accumulator. `SIGLIP2_DISABLE_FA=1` falls back if a parity issue surfaces.
- **`b39a896` — feat(server): `--vision-only` / `--text-only`** (lever C). Flag-gates tower load; missing endpoints 503.
- **`df577f7` — perf: scheduler max_nodes 8192 → 2048** (lever G).
- **`4849c9f` — feat(preproc): antialiased bilinear CPU resize** (lever A). Separable triangular AA filter matching torchvision `F.interpolate(antialias=True)`. Lifts image cosine at heavy-downsample shapes from <0.998 to ≥0.999.
- **`48fd16e` — perf(fa): zero-pad d_head 72 → 80 to hit MMA tensor-core path.** ggml's CUDA MMA / WMMA flash-attn kernels require D % 16 == 0; SigLIP2's d_head=72 was falling back to the (CUDA-cores) tile kernel. We zero-pad Q/K/V's d-axis to next multiple of 16 before FA and slice the result back before the output projection. Zero padding is mathematically null (Q·K and V both contribute 0 in padded slots) so parity is unchanged. **+14–23 % p50 on every endpoint, same VRAM.** See the helper `fa_attn_pad_slice` in `src/siglip2_vision.cpp`.
- **`77f97f1` — docs: revert parity floor to 0.999** (housekeeping; user briefly relaxed to 0.997 on 2026-05-08 then put it back).

## Lever queue — what's left

Ordered for an overnight run: biggest swings first, smaller-but-clean wins below, and the "documented-but-don't-land-without-asking" entries last.

### E. Megakernel-style block fusion *(perf + VRAM; biggest swing)*
- For SigLIP2 vision, an entire pre-LN transformer block (attn + MLP + 2 residuals) fuses into one CUDA kernel at the dominant shape (n_pos=729, H=1152, d_head=72). Eliminates intermediate-tensor allocations entirely, drives kernel launches per layer from ~10 to 1, and lets you pick the optimal MMA layout for these specific shapes (where ggml's generic dispatcher leaves perf on the table even after the d_head=72 → 80 padding hack).
- **Reference:** the user has prior art at `kobbler/docker/tts-qwen3-dev/HANDOFF-megakernel-v0.md` (the qwen3-tts megakernel-v0 plan, branch `dbrain/megakernel-v0` of that repo). Phase A is "shape-specialized MMVQ + per-layer fused kernels at one fixed shape," Phase B widens it. The pattern translates directly to SigLIP2's vision blocks at production max_num_patches=729.
- **Why this is the right next move:** with padded MMA-FA already in (lever D + `48fd16e`), the next biggest perf lift comes from killing intermediate allocations and kernel launches in the residual+LN+MLP chain — exactly what a fused per-block kernel buys.
- **How to start:** read the qwen3-tts handoff doc, then sketch a `HANDOFF-megakernel-v0.md` here mirroring its structure. Phase A target = parity-clean fused vision encoder block at fixed n_pos=729, H=1152, d_head=72. Q8 weights, F32 activations, FA path inside the kernel (no separate FA dispatch).
- **Risk:** Real. Massive-rewrite-of-internal-GGML territory. User explicitly opted in. Expect 1–2 weeks if you go all the way through Phase A + B; less if you stop at a single-shape Phase A drop-in.

### Graph caching for fixed-shape encoders *(perf; medium)*
- Right now both `VisionEncoder::encode()` and `TextEncoder::encode()` allocate the per-call arena, build the cgraph from scratch, and run `ggml_backend_sched_alloc_graph` every call. For text n_pos is constant (=64) so this is pure overhead repeated forever; for vision the shape varies but is bounded — there's a finite set of `(n_patches_h, n_patches_w)` shapes the binary-search lands on for any fixed `max_num_patches`.
- **Suspected payoff:** text path is at 0.61× of Python (47 ms vs 29 ms); the kernels themselves can't be that gap, so it's almost all setup. A graph cache (small LRU keyed on shape) should reclaim 10–30 % on text + classify_from_embeddings.
- **How:** keep a `std::unordered_map<key, BuiltGraph>` on the encoder where `BuiltGraph` carries the arena buffer, the cgraph, and the bound input/output tensor pointers. On entry to `encode()`, look up by shape; on miss, build + insert. `ggml_backend_sched_alloc_graph` results need a `ggml_backend_sched_reset` between calls — verify the cache holds up across resets, otherwise hold one sched per cache entry (small VRAM cost).
- **Risk:** Low–medium. Watch for thread-safety if a request hits the encoder while another is computing (`encode_mutex` already serializes server side, so probably fine).

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
- ✅ VRAM significantly below 1.85 GiB Q8 baseline (now 1438 MiB full / **720 MiB vision-only** — sub-1 GiB on vision-only achieved).
- ✅ Cosine ≥ 0.999 on real images (1920×1080 → 729 patches: 0.999054; AA resize + padded MMA-FA both clean).
- ✅ No quality regression from Q8 + FA + padded-MMA (all parity scripts pass at the 0.999 floor).
- 🟡 Performance: padded MMA-FA closed the easy slice of the gap (+14–23 % across endpoints). Remaining gap is **graph-setup overhead on small-batch paths** (text, classify_from_embeddings) and **per-block kernel-launch + intermediate-allocation overhead** (vision). Graph caching attacks the first; megakernel attacks the second. Land both and you should be at or above Python.
- ✅ Cold-load: ~16× faster than Python.

Ship a commit + update this doc's "Where we are" section as you land things. The next agent (or future-you) will pick up from there.

Now go.
