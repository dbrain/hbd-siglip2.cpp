# siglip2.cpp — productionization handoff

> ## Mood of the room
>
> Perf chapter is closed (see `HANDOFF-perf-next.md`'s "Final outcomes"
> section). The code is **functionally ready** to replace kobbler-vision-1
> on a 12 GB RTX 3060: faster, leaner VRAM, full endpoint parity for the
> kobbler-side callers. What's left is structural, not algorithmic.
>
> Five things to settle before this can ship:
> 1. **Where the source lives** — currently a feature branch on
>    `dbrain/qwen3-tts.cpp`, which is wrong for a separate product.
> 2. **The ggml fork** — same `dbrain/ggml@master` as qwen3-tts.cpp; OK to
>    keep shared, but document the shape so upstream-merge isn't a mystery.
> 3. **README + docs** — current README still says "early scaffolding";
>    needs a rewrite that matches reality.
> 4. **Production Docker image** — exists conceptually (kobbler's
>    `tts-qwen3` pattern is the template), needs to be built. Today's
>    iteration loop uses a builder image with bind-mount; prod needs a
>    self-contained image and a clean `compose.yml` switchover.
> 5. **Dev-vs-prod parity bench** — confirm the prod image runs at the same
>    perf as the iteration-loop builds.
>
> This document is the punch list for those five.

## 0. Current state of the world

### Where the code is right now

```
~/dev/siglip2.cpp                      ← directory
  remote: origin = git@github.com:dbrain/qwen3-tts.cpp.git
  branch: dbrain/siglip2-v0
  main:   "f88dc5f feat: initial Qwen3-TTS GGML implementation"

  ggml/                                ← submodule
    remote: origin = github.com/dbrain/ggml.git
    upstream:       github.com/ggerganov/ggml.git
    branch: dbrain/ggml@master
    HEAD:   4eec5550 (Q4_K get_rows kernel + megakernel hooks +
                      ggml_backend_cuda_init_with_priority)
```

So the working directory is a clone of `dbrain/qwen3-tts.cpp` with
`dbrain/siglip2-v0` checked out. `main` is the Qwen3-TTS implementation.
Siglip2 source coexists with Qwen3-TTS source on the same repo's separate
branch — they share `qwen3_tts::GGUFLoader`, `CMakeLists.txt`, the ggml
submodule, build helpers, and Docker iteration scripts.

### What's already built / shipping-ready

- **All four endpoints work and parity-pass**. See `HANDOFF-perf-next.md`
  "API gaps vs python kobbler-vision". Only gap is `return_last_hidden`
  which no kobbler caller uses.
- **Lifecycle is clean**: cold-start 1.33 s, unload 36 ms (frees 1506 MiB),
  reload 0.79 s, 400/400 concurrent classify pass.
- **Q8_0 production GGUF**: `models/siglip2-so400m-naflex-q8_0.gguf`
  (1.46 GB on disk). Plus a tokenizer model dependency:
  `reference/hf/siglip2-so400m-patch16-naflex/tokenizer.model`.
- **Build target list** (from `CMakeLists.txt`):
  - `siglip2-cli` (vision encode CLI)
  - `siglip2-text-cli` (text encode CLI)
  - `siglip2-server` ← the prod target
  - `siglip2-quantize` (HF→GGUF re-quantization tool)
  - `microbench_mmq` (perf gating harness, optional)

### Today's iteration-loop build

```bash
docker run --rm --gpus all -v $PWD:/src -w /src tts-qwen3-dev:builder \
  cmake --build build-cuda -j$(nproc)
```

(`tts-qwen3-dev:builder` is the kobbler iteration-loop image — has CUDA +
cmake + ggml prereqs.) The binary lands at `build-cuda/siglip2-server`.
Today's deploy is a `docker run` that bind-mounts the binary into a
runtime image — fine for iteration, **not appropriate for prod**.

## 1. Where the source should live

### Decision needed: separate repo or umbrella?

**Recommendation: fork to a dedicated repo.** Specifically:

```
github.com/dbrain/siglip2.cpp     ← new repo
  main branch carries everything currently on dbrain/siglip2-v0
  ggml submodule still points at github.com/dbrain/ggml.git (shared with qwen3-tts.cpp)
```

**Why not keep on the qwen3-tts.cpp branch:**

- Wrong namespace: `qwen3_tts::GGUFLoader` is reused from siglip2 code paths,
  which is misleading both ways.
- Tangled releases: a qwen3-tts.cpp tag/release doesn't mean siglip2 is
  ready, and vice-versa.
- Hard to sell to a contributor: "it's on a feature branch of a
  different project" is friction.
- The cleanup is straightforward — both projects already share ggml as a
  submodule; they don't actually share runtime libraries beyond that.

**Why not an umbrella repo (e.g. `dbrain/gguf-services` containing both
qwen3-tts + siglip2 + future):**

- The two projects have different release cadences, different build
  flag matrices, and different runtime images. An umbrella with two
  unrelated services means CI builds both on every commit (slow), and
  any shared code gets pulled into BOTH, even when only one needs it.
- llama.cpp + whisper.cpp pattern is two separate repos sharing ggml.
  That's the precedent.

### Rename plan (separate-repo path)

The grunt work is in three buckets:

1. **Namespace renames in source.** The code in `~/dev/siglip2.cpp/src/`
   is mostly already in `namespace siglip2 { ... }`. The exceptions are:
   - `qwen3_tts::GGUFLoader` (reused class) → either move to a `siglip2`
     namespace under siglip2.cpp, OR factor into a third `gguf_loader.cpp`
     library shared between the two projects. The first option is simpler
     and what I'd recommend (the loader is small; copies are cheap).
   - `qwen3_tts::init_preferred_backend`, `init_separate_backend`,
     `release_preferred_backend`, `load_tensor_data_from_file` — same call.

2. **CMakeLists rewrite.** The current `CMakeLists.txt` is the qwen3-tts
   one with siglip2 targets layered on top. Strip out everything not
   siglip2-relevant when copying to the new repo:
   - Drop `qwen3-tts-server`, `qwen3-tts-cli`, vocoder + speaker-encoder
     bits, voice-archive, etc.
   - Keep: ggml subdir, `siglip2_*` libs, `siglip2-cli`, `siglip2-text-cli`,
     `siglip2-server`, `siglip2_megakernel`, `microbench_mmq`,
     `siglip2-quantize`.
   - The CMake should be ~60 % smaller after the cull.

3. **Docs sweep.** Move `HANDOFF.md`, `HANDOFF-megakernel-v0.md`,
   `HANDOFF-megakernel-v0-phaseC-microbench.md`,
   `HANDOFF-megakernel-v0-phaseC-next.md`, `HANDOFF-perf-next.md`,
   `HANDOFF-prod.md` (this doc) into `docs/handoffs/` in the new repo.
   Rewrite `README.md` (see section 3 below).

### File-by-file checklist for the rename

When the agent doing the rename needs to know "which files belong to siglip2":

```
✅ KEEP (move to new repo):
   src/siglip2_*.{cpp,h}             # encoders, server, score, tokenizer, preproc
   src/cuda/siglip2_*.{cu,cuh,h,cpp} # megakernel + custom_mmq
   src/gguf_loader.{cpp,h}           # rename namespace qwen3_tts → siglip2
   bench/microbench_mmq.cpp          # already siglip2-only
   bench/microbench_cublaslt_helpers.{cu,cuh}
   scripts/parity_check_*.py
   scripts/bench_vs_python.py
   scripts/convert_siglip2_to_gguf.py
   tools/siglip2-quantize/main.cpp
   docs/                              # already siglip2-only
   reference/                         # HF model dir for tests
   models/                            # GGUFs (gitignored — fetch script needed)
   ggml                               # submodule — same dbrain/ggml.git remote
   AGENTS.md                          # repo-level agent instructions
   HANDOFF*.md                        # → docs/handoffs/
   README.md                          # full rewrite

❌ DROP:
   any qwen3-tts source (TTS server, vocoder, speaker_enc, voice archive,
   tokenizer-12hz, etc.) — none of it is siglip2's
   any qwen3-tts compose / Dockerfile bits
```

## 2. ggml fork — share or split

**Recommendation: keep shared.** Both projects use the same
`github.com/dbrain/ggml.git@master` submodule. The customizations are
generic (megakernel hooks are a hook system, not siglip2-specific):

- **Megakernel hook API** (`g_ggml_cuda_op_hook` + `g_ggml_cuda_graph_begin_hook`):
  generic per-op-dispatch hook in `ggml_backend_cuda_graph_compute`. Both
  qwen3-tts.cpp and siglip2.cpp register their own hooks via
  `ggml_cuda_set_op_hook` / `ggml_cuda_set_graph_begin_hook`. They don't
  conflict — process-global function pointers, latest registration wins.
  In practice, only one of the two services runs in any given process.
- **`ggml_backend_cuda_init_with_priority`**: a public API addition. Both
  projects use it now (qwen3-tts has talker=HIGH, siglip2 has vision=HIGH).
- **Q4_K get_rows GPU kernel**: qwen3-tts megakernel-v0 v9.6 added it.
  siglip2 only loads it via Plan A's K-padded GGUFs (off in prod). No
  conflict.
- **`GGML_CUDA_NO_MMA` debug knob**: present as uncommitted dirty content
  inside the submodule (intentional, debug-only). Same for both consumers.

**What this means for upstream tracking:** the `dbrain/ggml@master`
branch is N commits ahead of upstream `ggerganov/ggml.git` master. When
upstream releases a major refactor (e.g. ggml-cuda restructuring), the
fork-merger needs to:
1. Resolve conflicts in `ggml-cuda.cu` around the hook system.
2. Re-apply the priority API if upstream removed/renamed it.
3. Re-apply Q4_K get_rows if upstream's get_rows infrastructure changed.

These are all 1-2 hour merges, not multi-day. Keep the fork shared.

**If you ever want to split:** the cleanest cut would be to upstream the
hook API to ggerganov/ggml (it's generic and useful) and pull from
upstream master directly. Until then, shared fork is fine.

## 3. README rewrite

The current `README.md` (38 lines) opens with:

> **Status:** early scaffolding. Milestone 1 (vision-only fp16 parity) in
> progress.

That's stale by ~15 commits. It should now say: shipping-ready, all
endpoints at parity, perf table, build instructions, deploy instructions.

### Suggested new README sections

1. **What this is**: pure C++/GGML port of `google/siglip2-so400m-patch16-naflex`,
   drop-in for kobbler-vision's HTTP service.
2. **Why**: VRAM table (1542 MiB cpp vs 2322 MiB python), perf table
   (clean-GPU p50s for all 4 endpoints).
3. **Status table**: all 4 endpoints ✅, parity ≥ 0.999 on all 4 scripts,
   one feature gap (`return_last_hidden`) flagged.
4. **Quickstart** (3 commands):
   ```bash
   # Build the prod image (when section 4 lands):
   docker compose -f kobbler/docker-compose.yml build vision-cpp

   # Or iterate-loop build (today's path):
   docker run --rm --gpus all -v $PWD:/src -w /src tts-qwen3-dev:builder \
     cmake --build build-cuda -j$(nproc)

   # Run parity:
   ./scripts/run-parity.sh   # (helper script to write — see section 4)
   ```
5. **Architecture sketch**: pointer to the megakernel handoffs, mention
   the four phases (A0-A3 + B graph cache).
6. **Env knobs reference** (the table from `HANDOFF-perf-next.md`).
7. **Where to look for help**: `docs/handoffs/` for chronology.

Length target: ~150-200 lines. Drop the "Why" hand-wave from the current
README; let the perf numbers speak.

## 4. Production Docker image + kobbler integration

### Pattern: copy `kobbler/docker/tts-qwen3/Dockerfile`

The reference is `~/dev/kobbler/docker/tts-qwen3/Dockerfile` — a multi-
stage build that:
- **Stage 1 (builder)**: `nvidia/cuda:12.6.3-devel-ubuntu24.04`, builds
  `qwen3-tts-server` + the dbrain/ggml submodule with CUDA + flash-
  attention + CUDA graphs. ccache + apt cache mounts.
- **Stage 2 (runtime)**: `nvidia/cuda:12.6.3-runtime-ubuntu24.04`, copies
  the binary + ffmpeg shared libs, runs the binary directly as
  container CMD. No Python wrapper.

### What the siglip2-cpp Dockerfile needs

Drop `kobbler/docker/siglip2-cpp/Dockerfile` (or `kobbler-vision-cpp/`,
naming is a bikeshed) modeled on `tts-qwen3/Dockerfile`:

- **Builder stage**: pull dbrain/siglip2.cpp + dbrain/ggml submodule,
  `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 ...` (or
  `all-major` for portable images), produce `siglip2-server`.
- **Runtime stage**: nvidia/cuda runtime + the binary +
  `tokenizer.model` (the only Python-side dep is sentencepiece's
  protobuf model file, baked into the image at build time).
- **Model download**: at build time, fetch
  `siglip2-so400m-naflex-q8_0.gguf` from a known location (HF mirror
  via `huggingface-cli` or just `curl` from a pinned URL). Cache via
  Docker layer; bind-mountable to override.
- **Healthcheck**: `curl -sf http://localhost:8890/health` (matches
  python service exactly).
- **CMD**: `siglip2-server --model /models/siglip2-so400m-naflex-q8_0.gguf
  --tokenizer /models/tokenizer.model --port 8890`. Lazy-load + idle-
  unload via env (already supported in the binary:
  `LAZY_LOAD=true`, `IDLE_UNLOAD_SECONDS=N`).
- **CUDA arch arg**: `SIGLIP2_CUDA_ARCHS` build-arg, default `all-major`
  for portable images, `86` for RTX 3060-only builds. Same pattern as
  `tts-qwen3`.

### kobbler compose.yml change

The cleanest rollout is to add `vision-cpp` as a sibling service to the
existing `vision`, on a different port, then flip the kobbler API client
URL once the cpp service has been live for a day or two.

```yaml
# docker-compose.yml — addition
vision-cpp:
  build:
    context: ./docker/siglip2-cpp
    args:
      SIGLIP2_CUDA_ARCHS: "86"   # or all-major
  init: true
  restart: unless-stopped
  ports:
    - "8891:8890"                 # different host port; cpp listens on 8890
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  environment:
    NVIDIA_DRIVER_CAPABILITIES: compute,utility
    LAZY_LOAD: "true"
    IDLE_UNLOAD_SECONDS: "0"      # match python's behavior
  healthcheck:
    test: ["CMD", "curl", "-sf", "http://localhost:8890/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 30s             # cpp cold-start is 1.3 s, not 11 s
```

Then the kobbler API config (probably `kobbler/api/src/skip_detect/vision.rs`'s
`base_url` or a relevant config var) gets flipped from `http://vision:8890`
to `http://vision-cpp:8890`. Once flipped and confirmed stable for some
period, the python `vision:` service can be retired.

**If you want a hot-swap with no compose change**: keep the cpp service
named `vision` and rebuild + recreate. Python image gets dropped from
docker images. Cleaner long-term but riskier rollback.

### Build pipeline parity (dev vs prod)

The current `tts-qwen3-dev:builder` builder is a 12 GB iteration-loop image.
Production should NOT use the dev builder for shipping artifacts — that
image has tools and caches that don't belong in releases. Instead:

- **Dev iteration loop**: keep using `tts-qwen3-dev:builder` for fast
  rebuilds (ccache, bind-mount). Outputs a binary that you then test
  against the dev runtime image.
- **Prod build**: `docker compose build vision-cpp` runs the multi-stage
  Dockerfile. Slower but produces a sealed image.
- **Parity check**: after the prod image is built, run the perf bench
  AGAINST the prod image (not the iteration-loop binary) and confirm
  numbers match `HANDOFF-perf-next.md`'s "Final outcomes" table within
  ~5 %. This is the dev-vs-prod parity gate.

Concretely the parity bench is:
```bash
# Stop iteration-loop containers, kobbler-vision-1, etc.
# Bring up only the prod siglip2 image:
docker compose up -d vision-cpp
# Wait for /health → model_loaded:true
# Run scripts/bench_vs_python.py with --cpp-url http://localhost:8891
# Numbers should be: classify p50 ~57 ms, embeddings ~44 ms, etc.
```

If the prod image is slower than the iteration-loop binary, common
causes:
- `CMAKE_CUDA_ARCHITECTURES=all-major` adds PTX overhead (15-20 MiB
  resident per arch). For a single-GPU deploy, pin to the actual arch.
- `ccache` was disabled and gcc/nvcc compiled at lower opt level. Check
  build logs for `-O3`.
- Different CUDA runtime version between builder and runtime stage.
  Pin both to the same `12.6.3-*-ubuntu24.04`.

## 5. Rollout order — the punch list

1. **Fork** `dbrain/siglip2-v0` → `github.com/dbrain/siglip2.cpp` (new
   repo). Cull the qwen3-tts source (section 1's checklist). Push.
2. **README rewrite** in the new repo (section 3). Drop `HANDOFF*.md`
   into `docs/handoffs/`. Push.
3. **Build the prod Docker image** at
   `kobbler/docker/siglip2-cpp/Dockerfile` (section 4). Local build first,
   verify `/health` + one `/v1/classify`.
4. **Run dev-vs-prod parity bench** (section 4). Compare against
   `HANDOFF-perf-next.md` final outcomes. Tolerance ~5 % on p50.
5. **Add `vision-cpp:` service** to `kobbler/docker-compose.yml`
   (section 4 yaml). Deploy alongside `vision:`. Run end-to-end smoke
   test of the kobbler API path (`skip_detect` is the main consumer).
6. **Flip the kobbler API client** to use `vision-cpp`. Watch logs and
   metrics for ~24-48 h.
7. **Retire the python `vision:` service**: drop from compose, prune
   the kobbler-vision Docker image.

## What I'd tell future-you

- The hard work — getting the cpp service to feature-parity, performance,
  and stability — is done. What's left is mechanical (rename, rebuild,
  flip).
- The biggest single risk is **CUDA arch mismatch** between the builder
  image and the runtime image. The dev iteration loop hides this because
  builder = runtime. Production splits them; double-check compute
  capability detection survives the split.
- The second biggest risk is **the persistent worker thread + private
  backend interaction** under unusual load patterns. The perf-next
  handoff has receipts for 400 concurrent classify and 3× unload+reload
  cycles, but rare patterns (e.g. /unload during a 30-image batch) might
  surface things. The unload code is correct (server-side timing
  verified), but client behavior under partial completion deserves a
  smoke test.
- **Do not** try to chase further perf wins until you've bounced this off
  a non-Ampere GPU. Several of the aborted targets in `HANDOFF-perf-next.md`
  ("dead on Ampere") might revive on sm89+ where cuBLASLt int8 supports
  per-tile scales and the tile FA kernel is reportedly tuned. The
  re-test harnesses (`microbench_mmq`'s cuBLLt column,
  `SIGLIP2_DISABLE_FA_PAD` env) are in tree for exactly this.

Now go.
