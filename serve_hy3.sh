#!/usr/bin/env bash
# Hy3-295B (Tencent Hunyuan 3, 295B MoE / 21B active) NVFP4-W4A16 on 2x RTX PRO 6000
# Blackwell (sm_120), vLLM 0.24.0 venv. Adapted from tonyd2wild/Hy3-295B-NVFP4-MTP-2x-DGX-Spark
# (2x DGX-Spark GB10 2-node ray -> single-node 2-GPU mp executor here).
#
# Quant is MARLIN-kernel-only by design (avoid FlashInfer native-FP4 path). MTP layer preserved
# for optional spec decoding. 8 KV heads -> TP in {1,2,4}. 181GB weights on 2x96GB is TIGHT
# (~90.5GB/card) -> high util, modest ctx, fp8 KV, enforce-eager.
set -uo pipefail
VENV=/home/user/vllm-venv
: "${MODEL:=/home/user/models/Hy3-NVFP4-W4A16}"
: "${LOG:=/home/user/models/vllm_hy3.log}"
: "${CTX:=32768}"
: "${UTIL:=0.96}"          # weights ~90.5GB/card of 95.6 -> push high; leaves ~few GB for fp8 KV
: "${TP:=2}"
: "${MAXSEQS:=4}"
: "${MAXBATCH:=8192}"
: "${PORT:=8000}"
: "${EAGER:=1}"            # repo: eager WINS on this stack (cudagraphs cost ~25%)
: "${KV:=fp8_e4m3}"       # fp8_e4m3 | auto
: "${SPEC:=0}"            # 0=off | mtp  (mtp spec-1; lossless, repo says spec-1 beats spec-2 on Blackwell)
: "${SPECTOK:=1}"
: "${PARSERS:=1}"         # 1 = enable hy_v3 tool+reasoning parsers (needs :opensource sed patch applied)
: "${THINK:=0}"          # 0=off (template default no_think) | high | low  -> force reasoning_effort server-side
: "${SERVED:=hy3}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=12.0
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP4=0   # force marlin W4A16 path (flashinfer FP4 freezes/gaps on Blackwell)
export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$PATH"

args=( serve "$MODEL"
  --served-model-name "$SERVED"
  --trust-remote-code
  --tensor-parallel-size "$TP"
  --disable-custom-all-reduce
  --max-model-len "$CTX"
  --gpu-memory-utilization "$UTIL"
  --max-num-seqs "$MAXSEQS"
  --max-num-batched-tokens "$MAXBATCH"
  --kv-cache-dtype "$KV"
  --host 0.0.0.0 --port "$PORT" )
[ "$EAGER" = "1" ] && args+=( --enforce-eager )
[ "$PARSERS" = "1" ] && args+=( --enable-auto-tool-choice --tool-call-parser hy_v3 --reasoning-parser hy_v3 )
[ "$SPEC" = "mtp" ] && args+=( --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPECTOK}" )

echo "[start $(date +%H:%M:%S)] KV=$KV SPEC=$SPEC PARSERS=$PARSERS CTX=$CTX UTIL=$UTIL EAGER=$EAGER" | tee -a "$LOG"
echo "CMD: $VENV/bin/vllm ${args[*]}" | tee -a "$LOG"
exec "$VENV/bin/vllm" "${args[@]}" >> "$LOG" 2>&1
