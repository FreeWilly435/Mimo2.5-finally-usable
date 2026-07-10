# Mimo2.5-finally-usable

A recipe to have **MiMo‑V2.5 finally usable** on 2× RTX PRO 6000 — with decent **context /
quality / speed** and **all the modalities activated** (text, vision, audio).

The thing that makes it work is an **NVFP4 KV cache**. That single change — 4‑bit KV instead of
bf16 — is what turns a cramped 64K‑context server into a **512K‑context** one on the same VRAM,
while the model still scores **88/100** on a hard agentic tool‑calling benchmark, serves at
**~128 tok/s**, transcribes audio, and reads images.

> **TL;DR** — MiMo‑V2.5‑DFlash‑**MXFP4A16** weights + **NVFP4 KV cache** + a 2‑line
> backend‑selector patch → a 512K‑context, 88/100, omni‑modal daily driver on 2×96 GB Blackwell.

---

## The model

- **MiMo‑V2.5‑DFlash‑MXFP4A16** (Xiaomi MiMo‑V2.5), served as `MiMo-V2.5`. 310B total / ~15B active MoE.
  Served weights: [`chriswritescode/MiMo-V2.5-DFlash-MXFP4A16`](https://huggingface.co/chriswritescode/MiMo-V2.5-DFlash-MXFP4A16)
  — the MXFP4A16 build of [`XiaomiMiMo/MiMo-V2.5-DFlash`](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-DFlash).
- Mixed precision (the `config.json` under‑describes it):
  - **MoE experts:** MXFP4 (U8‑packed 4‑bit + E8M0 block scales, block 32)
  - **`qkv_proj`:** block‑FP8 (`F8_E4M3`) · **`o_proj`:** BF16
  - Attention geometry: `heads=64, kv_heads=4, head_dim=192, v_head_dim=128` → **DiffKV** (K width ≠ V width)
- **Hybrid attention** — of 48 layers, only ~**9 are full‑attention**; the other **39 are
  sliding‑window (128)** (`attention_chunk_size=128`). Only the 9 full layers' KV grows with
  sequence length, which is *why* half‑million‑token context is affordable at all.
- **Omni‑modal:** despite `architectures: [MiMoV2ForCausalLM]`, the checkpoint carries a ViT
  tower + `audio_encoder` + `speech_embeddings`. Load it as `MiMoV2OmniForCausalLM` for image
  **and** audio input.
- Native context ceiling: `max_position_embeddings = 1048576`, `rope_theta = 1e7` (no YaRN).

## Hardware

| | |
|---|---|
| GPUs | 2× NVIDIA RTX PRO 6000 Blackwell, 96 GB each (one Workstation 350 W, one Max‑Q 325 W), sm_120 |
| Link | PCIe, **no NVLink**, separate PCIe root ports |
| Parallelism | Tensor‑parallel 2 |
| Desktop | moved onto the onboard BMC GPU (`nvidia_drm modeset=0`) so both NVIDIA cards are fully free for the model |

---

## Environment & versions

**Host** — Ubuntu 25.10 · kernel 6.17.0‑40 · NVIDIA driver **595.71.05** · CUDA toolkit 13.3 ·
Docker 29.1.3 · 2× RTX PRO 6000 Blackwell (sm_120).

Each model runs in its own stack (exact pins, so results are reproducible):

| stack | vLLM | torch | CUDA | FlashInfer | transformers | triton | python |
|---|---|---|---|---|---|---|---|
| **MiMo — this recipe** | `0.11.2.dev279` (eldritch, cu132) | `2.12.0+cu132` | 13.2 | `0.6.13` | `5.12.1` | `3.7.0` | 3.12.3 |
| DeepSeek‑V4‑Flash‑DSpark | `0.11.2.dev279` (eldritch, cu132) | `2.12.0+cu132` | 13.2 | `0.6.13` | `5.13.0` | `3.7.0` | 3.12.3 |
| Hunyuan‑3 (Hy3) | `0.24.0` | `2.11.0+cu130` | 13.0 | `0.6.12` | `5.9.0` | `3.6.0` | 3.12.13 |

MiMo additionally carries **`torchaudio 2.11.0+cpu`** (audio — the CPU wheel, see [gotchas](#gotchas-we-hit))
and `torchvision 0.27.0+cu132`; `numpy 2.3.5`.

**Base images / envs:**
- MiMo: `cstechdev/vllm:eldritch-enlightenment-mimo25-dflash-mxfp4-v2-cu132-20260705` → patched to
  `mimo-mxfp4-nvfp4:local` (see [`Dockerfile.mimo-mxfp4-nvfp4`](./Dockerfile.mimo-mxfp4-nvfp4)).
- DSpark: `voipmonitor/vllm:eldritch-enlightenment-v45c1582-b12xf3686b5-pc1441b5-cu132-20260704`.
- Hy3: native venv (vLLM 0.24.0) — no container.

---

## ⭐ The NVFP4 KV cache (the unlock)

Storing the KV cache in **NVFP4** (`--kv-cache-dtype nvfp4`) instead of bf16/fp8 is the single
biggest lever. At ~4.5 bits/value it roughly **quarters** KV memory vs bf16 — the difference
between a 64K server and a **512K** one on identical hardware. NVFP4 KV pool at GMU 0.96:
**632,910 tokens** — enough to hold a full 512K‑token request at **1.21× concurrency**, with
~2.4 GiB/card of prefill headroom to spare.

**The kernel was never the problem.** The Triton **DiffKV** attention backend
(`triton_attn_diffkv.py`) already handles NVFP4 — it's byte‑identical to the one our other server
uses to serve NVFP4 daily. The stock vendor image simply **never *declared*** NVFP4 as supported,
and (unlike a hand‑constructed backend) routes backend selection through vLLM's selector, which
enforces `supported_kv_cache_dtypes`. So the fix is a **2‑line patch** baked into the image
([`Dockerfile.mimo-mxfp4-nvfp4`](./Dockerfile.mimo-mxfp4-nvfp4)):

1. **Declare it** — add `"nvfp4"` to `supported_kv_cache_dtypes` in `triton_attn_diffkv.py`.
2. **Don't double‑pad V** — exclude the DiffKV backends from the `pad_value_for_fa` path in
   `mimo_v2.py`, or the allocator builds a **384‑wide** cache (192 K + 192 *padded* V) while the
   DiffKV kernel reads **320** (192 K + 128 *packed* V) →
   `RuntimeError: shape '[…,320]' is invalid for input of size … (== 384‑wide)`.

**Speculation is deliberately OFF.** MiMo ships a 7‑token **DFlash** drafter, but here it measured
*worse* (−28 % single‑stream, and it reserves KV that shrinks the pool ~56 %). It's also
incompatible with NVFP4 (the draft head routes through the selector, and the only NVFP4‑capable
attention backend rejects attention sinks) — so NVFP4 forces spec off, at no cost.

## The serve config

`--kv-cache-dtype nvfp4 --block-size 32 --attention-backend TRITON_ATTN_DIFFKV`,
`moe_backend=flashinfer_cutlass`, `linear_backend=b12x`, **TP2**, **`--gpu-memory-utilization 0.96`**,
**`--max-model-len 524288`** (512K), **no speculation**, cudagraphs `FULL_AND_PIECEWISE`,
`--async-scheduling`. Full launcher: [`serve_mimo_mxfp4_nvfp4.sh`](./serve_mimo_mxfp4_nvfp4.sh).

> **Why 0.96 and not higher:** GMU **0.98** boots with a bigger (816K) KV pool but is a **trap** —
> vLLM's memory profiler under‑reserves activation memory (flashinfer allocates workspace lazily),
> so the *first real prefill* OOMs and kills the engine
> (`Tried to allocate 526 MiB, 291 MiB free`). 0.96 leaves ~2.7 GiB/card of prefill headroom and
> serves long context **reliably**. True 1M context needs ~2.5 GiB more KV than exists on 2×96 GB.

### This recipe vs. the upstream default

The model card ships a reference recipe built for a **4‑GPU** box (image + full `docker run` at
[`chriswritescode/MiMo-V2.5-DFlash-MXFP4A16`](https://huggingface.co/chriswritescode/MiMo-V2.5-DFlash-MXFP4A16)).
If you have 4× 96 GB, just run that — it's simpler. This repo is what you change to run the **same
model and ~the same 500K context on half the GPUs**:

| knob | Upstream card (4‑GPU) | This recipe (2‑GPU) |
|---|---|---|
| Parallelism | **TP4** (`CUDA_VISIBLE_DEVICES=0,1,2,3`) | **TP2** |
| **KV cache** | `auto` (bf16) | **`nvfp4`** — the change that makes 512K fit on 2 cards |
| Attention backend | `TRITON_ATTN` | `TRITON_ATTN_DIFFKV` + the 2‑line declare/pad patch |
| **DFlash speculation** | **on**, 7 tokens | **off** — worse on this box, and nvfp4‑incompatible |
| GPU‑mem‑util | 0.90 | 0.96 |
| max‑num‑seqs / batched‑tokens | 64 / 16384 (throughput) | 4 / 4096 (long‑ctx single‑stream) |
| block‑size | 64 | 32 |
| max‑model‑len | 500,000 | 524,288 (512K) |
| image | `…mxfp4-v1-cu132-20260705` | `…mxfp4-v2-…` + patch → `mimo-mxfp4-nvfp4:local` |
| audio | — | `OMNI=1` (MiMoV2Omni) |

**Why the differences chain together:** on TP4 the weights are ~40 GB/card, leaving ample room for
a bf16 KV cache *and* DFlash's scratch — so the upstream recipe keeps both and cranks concurrency
(64 seqs). On **TP2** the weights are ~83 GB/card with only ~13 GB left; a bf16 KV cache can't hold
long context there. Quantizing KV to **NVFP4** (~4.5 bits/value) reclaims it — but that path runs
through the **DiffKV** backend (hence the 2‑line patch), and the only nvfp4‑capable attention
backend rejects the attention sinks the DFlash draft head needs, so **speculation comes off**. Net
trade: give up the upstream's raw throughput, gain long context + audio on **two** cards instead of
four.

---

## Quality — hard agentic tool‑calling benchmark

`tool-eval-bench` has two hard settings, and they are **not** the same test:
- `--hardmode` — the full **84‑scenario** suite (all categories, including the hard "Category P").
- `--hardmode-only` — **only the 15 hardest** (Category P); a stress subset where every model scores lower.

**On the full `--hardmode`, MiMo‑V2.5‑MXFP4A16 (this recipe) = 88/100 ★★★★** (67 pass / 13 partial /
4 fail, greedy temp 0) — parity with its NVFP4 predecessor (89), and the reason it earns the daily‑driver slot.

For a **model‑vs‑model** head‑to‑head, the common basis is the **15‑hardest subset, greedy (temp 0),
`--parallel 1`** — the bench's own recommendation, because running scenarios concurrently causes
saturation timeouts that get scored as FAIL and unfairly sink the slower models:

| Model — `--hardmode-only` (15 hardest, greedy, parallel 1) | score | notes |
|---|---:|---|
| **DeepSeek‑V4‑Flash‑DSpark** (MXFP4 + DeepSpec) | **83** | strongest on the hardest scenarios (spec‑decode + tuning) |
| **Hunyuan‑3 (Hy3)** NVFP4‑W4A16 | **73** | greedy; 60–77 at the card's temp 0.9 (noisy) |
| **MiMo‑V2.5‑MXFP4A16** *(this recipe)* | **70** | greedy — but **88** on the full 84‑scenario `--hardmode` |
| MiMo‑V2.5‑NVFP4 (previous driver) | 69.7 | temp 0.7 / top‑p 0.8 ×3 (older run) |

Read it honestly: on the **15 hardest** scenarios the three big MoEs cluster **70–83, with DSpark ahead**.
MiMo's edge isn't the hard subset — it's the **broad** suite (88), plus 512K context, audio + vision, and
~128 tok/s: a well‑rounded daily driver rather than a hard‑subset champion.

> **On Hy3 specifically:** an earlier pass scored it 67–68, but that used `--parallel 4` (saturation) and
> temp 0.7. Re‑run properly — `--parallel 1`, the card's temp 0.9 / top_p 1.0 — it's noisy (60/77), and
> **greedy it's a clean 73**. A real, if modest, improvement over the old number — and the failures are
> genuine quality misses, not timeouts.

**Source models:** MiMo‑V2.5 → served weights
[`chriswritescode/MiMo-V2.5-DFlash-MXFP4A16`](https://huggingface.co/chriswritescode/MiMo-V2.5-DFlash-MXFP4A16)
(the MXFP4A16 build of [`XiaomiMiMo/MiMo-V2.5-DFlash`](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-DFlash)) ·
Hunyuan‑3 → [`tencent/Hy3`](https://huggingface.co/tencent/Hy3),
NVFP4 quant [`kodelow/Hy3-NVFP4-W4A16`](https://huggingface.co/kodelow/Hy3-NVFP4-W4A16) ·
DeepSeek‑V4‑Flash → [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai) (DSpark variant).

### How the comparison models were served

All three run on the **same 2× RTX PRO 6000** box, TP2, no NVLink — so every config carries
`--disable-custom-all-reduce` + `NCCL_P2P_DISABLE=1` (separate PCIe root ports).

**DeepSeek‑V4‑Flash‑DSpark** — MXFP4, ~156 GB
- Image `voipmonitor/vllm:eldritch-enlightenment-…cu132-20260704`; **b12x** kernels
  (`--attention-backend B12X_MLA_SPARSE --moe-backend b12x --linear-backend b12x`) + a patched
  NCCL (`LD_PRELOAD` local `libnccl`).
- `--kv-cache-dtype fp8 --block-size 256`, TP2, GMU 0.95, up to `--max-model-len 819200` (~800K),
  cudagraphs `FULL_AND_PIECEWISE`.
- **DeepSpec drafter on**: `--speculative-config {method: dspark, num_speculative_tokens: 5,
  draft_sample_method: probabilistic}` → ~300 tok/s structured. `reasoning_effort=high`.
- Parsers: `deepseek_v4` tokenizer / tool / reasoning.
- **Full script:** [`serve_dsv4_dspark.sh`](./serve_dsv4_dspark.sh) (env knobs `KV_DTYPE`,
  `DSPARK_TOKENS`, `REASONING_EFFORT`, `MAX_MODEL_LEN`, `BACKEND`).

**Hunyuan‑3 (Hy3) 295B** — NVFP4‑W4A16, ~181 GB (~90.5 GB/card, tight)
- vLLM 0.24 (native venv, sm_120). Quant is **Marlin‑W4A16 only** by design
  (`VLLM_USE_FLASHINFER_MOE_FP4=0` — the FlashInfer FP4 path stalls on Blackwell).
- `--kv-cache-dtype fp8_e4m3`, TP2, GMU 0.96, `--max-model-len 32768`, **`--enforce-eager`**
  (cudagraphs cost ~25 % on this stack); MTP spec available but off.
- Parsers: `hy_v3` tool + reasoning (needs a small `:opensource` special‑token sed patch).
- Sampling: card‑recommended **temp 0.9 / top_p 1.0**, `reasoning_effort` default `no_think`.
- **Full script:** [`serve_hy3.sh`](./serve_hy3.sh) (env knobs `KV`, `CTX`, `UTIL`, `SPEC`, `THINK`, `PARSERS`).

(MiMo's own launcher is [`serve_mimo_mxfp4_nvfp4.sh`](./serve_mimo_mxfp4_nvfp4.sh); config summarized in
[The serve config](#the-serve-config) above.)

---

## Long context — needle‑in‑a‑haystack

A unique fact — a plainly‑stated counted quantity — is buried at depths 10 / 50 / 90 % in a filler
haystack of growing length; the model must retrieve it. Context sizes are read back from
`usage.prompt_tokens`. Harness: [`niah.py`](./niah.py) (see the behavior note below for why the
needle is a neutral count rather than a "secret").

Retrieval was **100 %** — every needle found *and* echoed in the final answer, at every depth, up
to the largest context tested (~500K, just under the 512K window). Single‑stream, and the server
stayed healthy across six back‑to‑back ~500K prefills — confirming 512K serves reliably at GMU 0.96.

| Context (`prompt_tokens`) | depth 10 % | depth 50 % | depth 90 % | prefill |
|---|:---:|:---:|:---:|---|
| ~7,900   | ✅ | ✅ | ✅ | ~1–2 s |
| ~31,900  | ✅ | ✅ | ✅ | ~4–6 s |
| ~127,900 | ✅ | ✅ | ✅ | ~40 s |
| ~255,900 | ✅ | ✅ | ✅ | ~132 s |
| ~499,900 | ✅ | ✅ | ✅ | ~455 s |

**A behavior note worth flagging.** The first pass used a needle phrased as a *"secret passcode."*
Up to 128K the model returned it fine, but at **256K–500K it started *refusing*** — *"I'm not going
to provide that…"* — even though the value was plainly present in its reasoning trace. It had
**retrieved** the needle but treated echoing a "secret" as unsafe at long range. Swapping to a
neutral needle (a counted quantity) removed the refusals and retrieval was perfect (table above).
So long‑context retrieval here is solid; the only "misses" were a safety artifact of the wording.

---

## Multimodal — audio + vision

**Audio input works.** Feeding `/usr/share/sounds/alsa/Front_Center.wav` (a spoken "front,
center") via the OpenAI `input_audio` content type, the model transcribed it as **"Front center"**
— the WAV is decoded, mel‑spectrogrammed, and attended over end‑to‑end. Harness:
[`audio_test.py`](./audio_test.py). One gotcha: MiMo is a *reasoning* model, so give it enough
`max_tokens` (≥ ~200) or the visible answer comes back empty (the budget is spent on hidden
reasoning).

Vision works in the same `OMNI=1` mode (the ViT tower loads; verified describing photos and
reading split images). Audio needs **torchaudio** in the image — the **CPU** wheel, because the
PyPI CUDA wheel's `_check_cuda_version()` aborts import against this build's `torch 2.12.0+cu132`
(the mel spectrogram is pure `torch.stft`, CPU‑side, so no CUDA is needed). See Dockerfile layer 4.

---

## Reproduce

**MiMo — this recipe (the daily driver):**
```bash
# 1) build the patched image — EMPTY build context on purpose (see gotchas)
sudo docker build -f Dockerfile.mimo-mxfp4-nvfp4 -t mimo-mxfp4-nvfp4:local /some/empty/dir
# 2) serve (a systemd unit runs this in the foreground; OMNI=1 enables audio)
GMU=0.96 MAX_MODEL_LEN=524288 OMNI=1 ./serve_mimo_mxfp4_nvfp4.sh
```

**The two comparison models** (each needs both GPUs, so run one at a time):
```bash
# DeepSeek-V4-Flash-DSpark (docker; b12x kernels + DeepSpec drafter)
./serve_dsv4_dspark.sh                     # knobs: KV_DTYPE / DSPARK_TOKENS / REASONING_EFFORT / MAX_MODEL_LEN

# Hunyuan-3 / Hy3 (native vLLM 0.24 venv; Marlin-W4A16, enforce-eager)
#   first apply the :opensource special-token sed patch to the hy_v3 parsers (see script header)
KV=fp8_e4m3 CTX=32768 UTIL=0.96 PARSERS=1 ./serve_hy3.sh
```

**The benchmark** (same tool for every number in the table):
```bash
# fair model-vs-model basis: the 15 hardest, sequential (no saturation), greedy
tool-eval-bench --model <served-name> --base-url http://localhost:8000 --backend vllm \
    --hardmode-only --temperature 0 --top-p 1.0 --parallel 1 --json --json-file out.json
# MiMo's headline 88 uses the FULL suite instead: --hardmode  (84 scenarios)
```

Files: [`serve_mimo_mxfp4_nvfp4.sh`](./serve_mimo_mxfp4_nvfp4.sh) ·
[`Dockerfile.mimo-mxfp4-nvfp4`](./Dockerfile.mimo-mxfp4-nvfp4) ·
[`serve_dsv4_dspark.sh`](./serve_dsv4_dspark.sh) · [`serve_hy3.sh`](./serve_hy3.sh) ·
[`niah.py`](./niah.py) · [`audio_test.py`](./audio_test.py)

## Gotchas we hit

- **NVFP4 KV needs the 2‑line selector patch** — the kernel already supports it; the image just
  didn't declare it.
- **GMU 0.98 is a boot‑but‑OOM trap** with long context — the profiler under‑reserves activation
  memory and the first prefill dies. Keep GMU ≤ 0.96 and re‑test with a *real* long prefill, not
  just a successful boot.
- **`--max-model-len` in the `sh -lc` wrapper** must be expanded by the *outer* shell (`'"$VAR"'`),
  or it reaches the container empty → `invalid human_readable_int_or_auto value: ''`.
- **torchaudio**: install the **CPU** wheel; the CUDA wheel mismatches `torch 2.12+cu132`.
- **Don't `rm /dev/shm/psm_*`** while another vLLM runs — it kills that engine's IPC queues
  (`RuntimeError: cancelled`).
- **Docker build context**: build from an *empty* dir; the model tree is ~1.6 TB and will fill the
  disk if used as context.

*Measured 2026‑07‑10 on the box above (vLLM v0.11.2.dev279, base image
`cstechdev/vllm:eldritch-enlightenment-mimo25-dflash-mxfp4-v2-cu132-20260705`).*
