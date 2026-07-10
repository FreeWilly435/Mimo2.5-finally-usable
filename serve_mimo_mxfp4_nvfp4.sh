#!/usr/bin/env bash
# DAILY DRIVER: MiMo-V2.5-DFlash-MXFP4A16 + nvfp4 KV cache, TP2, on :8000.
# Runs in FOREGROUND so systemd (mimo-mxfp4.service) owns the lifecycle.
#
# Measured 2026-07-10 (tool-eval-bench hardmode: 88/100 ★★★★):
#   KV pool 361,204 tokens (5.51x concurrency @64K)  — 1.94x the old 0703 driver
#   127.82 tok/s conc=1   |   331.60 tok/s conc=4
#
# WHY SPECULATION IS OFF: DFlash (7 tokens) measured *worse* on this box —
#   auto KV, spec ON : 191,575 tok pool, 101.78 / 360.99 tok/s
#   auto KV, spec OFF: 299,361 tok pool, 130.58 / 339.70 tok/s
# i.e. +28% conc=1 and +56% KV pool by disabling it. It is also *incompatible*
# with nvfp4: the DFlash draft head (head_dim=128, use_non_causal=True) goes
# through the attention selector, and the only nvfp4-capable backend
# (flashinfer) rejects attention sinks. So nvfp4 REQUIRES spec off.
#
# MULTIMODAL: despite `architectures: [MiMoV2ForCausalLM]`, this checkpoint IS
# vision-capable — config carries vision_config + image_token_id(151655) +
# video_token_id + audio_config, and the ViT tower loads
# ("Using AttentionBackendEnum.FLASH_ATTN for MMEncoderAttention").
# Verified 2026-07-10: correctly described a real photo, named a drawn circle,
# and read the top band of a split image. (Audio path untested.)
# Caveat: a SOLID-COLOUR fill image gets misnamed (uniform patches, no spatial
# variance) — that is a degenerate input, not a broken vision tower.
#
# Roll back to the old omni driver with:
#   sudo systemctl disable --now mimo-mxfp4 && sudo systemctl enable --now mimo-0703
set -eu

IMG=${IMG:-mimo-mxfp4-nvfp4:local}
NAME=${NAME:-mimo-mxfp4}
MODEL_DIR=${MODEL_DIR:-/home/user/models/MiMo-V2.5-DFlash-MXFP4A16}
# 2026-07-10: the GNOME desktop was moved OFF the NVIDIA cards onto the ASPEED BMC
# (nvidia_drm modeset=0 + mutter primary-gpu udev rule; see POST_REBOOT_desktop_bmc.md),
# so both cards are symmetric and there's no desktop VRAM to contend with.
# BUT: GMU 0.98 is a TRAP with long context — it boots (816k KV pool) yet the memory
# profiler under-reserves activation memory (flashinfer allocates workspace lazily), so the
# FIRST real prefill OOMs and kills the engine ("Tried to allocate 526 MiB, 291 MiB free").
# 0.96 leaves ~2.7 GiB/card of prefill headroom and serves long context reliably. Do NOT raise
# past ~0.96 unless you also shrink MAX_MODEL_LEN and re-test with a real long prefill.
GMU=${GMU:-0.96}

# MAX_MODEL_LEN: per-request context window. The checkpoint's native max is 1048576 (1M),
# but at GMU 0.98 the NVFP4 KV pool is 9.24 GiB and a single 1M request needs 11.71 GiB, so
# Per-request context window. The checkpoint's native max is 1048576 (1M), but usable context is
# bound by BOTH the NVFP4 KV pool AND prefill activation headroom (see the GMU note above):
#   GMU 0.98 -> 816k KV pool, but OOMs on the first prefill (unusable).
#   GMU 0.96 -> ~643k KV pool + ~2.7 GiB headroom -> 512K serves reliably (verified w/ NIAH).
# 524288 (512K) = 8x the original 64K, block-aligned, pool holds it at 1.25x concurrency. Only
# ~9 of 48 layers are full-attention (39 are sliding-window/128), which is what makes 512K fit.
# 1M is genuinely out of reach on 2x96GB. To push toward ~600K, try GMU 0.965 and RE-TEST a real
# long prefill (do not trust boot alone — 0.98/800K booted then crashed).
MAX_MODEL_LEN=${MAX_MODEL_LEN:-524288}

# OMNI=1 loads MiMoV2OmniForCausalLM (adds the audio_encoder + speech_embeddings towers ->
# audio input). Verified 2026-07-10: audio needs torchaudio (baked into the image, layer 4
# of the Dockerfile). Audio towers cost ~0 KV here. Set OMNI=1 in the systemd unit.
OMNI=${OMNI:-0}
OMNI_ARGS=''
if [ "$OMNI" = "1" ]; then
  OMNI_ARGS='--hf-overrides "{\"architectures\":[\"MiMoV2OmniForCausalLM\"]}" --limit-mm-per-prompt "{\"image\":2,\"video\":0,\"audio\":1}"'
fi

exec docker run --rm --name "$NAME" \
  --gpus all --ipc=host --shm-size=32g --init \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
  -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUTE_DSL_ARCH=sm_120a \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=1 \
  -v "$MODEL_DIR":/model:ro \
  --entrypoint /bin/sh \
  "$IMG" \
  -lc 'exec vllm serve /model \
    --served-model-name MiMo-V2.5 \
    --trust-remote-code \
    '"$OMNI_ARGS"' \
    --host 0.0.0.0 --port 8000 \
    --kv-cache-dtype nvfp4 \
    --block-size 32 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization '"$GMU"' \
    --max-num-seqs 4 \
    --max-num-batched-tokens 4096 \
    --max-model-len '"$MAX_MODEL_LEN"' \
    --max-cudagraph-capture-size 128 \
    --attention-backend TRITON_ATTN_DIFFKV \
    --kernel-config.moe_backend flashinfer_cutlass \
    --kernel-config.linear_backend b12x \
    --reasoning-parser mimo \
    --tool-call-parser mimo \
    --enable-auto-tool-choice \
    --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}" \
    --async-scheduling \
    --no-scheduler-reserve-full-isl \
    --enable-chunked-prefill \
    --enable-prefix-caching'
