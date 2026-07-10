#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4-Flash-DSpark on 2x RTX PRO 6000 Blackwell (SM120), TP2.
# Based on local-inference-lab/rtx6kpro ds4dspark-v9.

IMAGE="${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v45c1582-b12xf3686b5-pc1441b5-cu132-20260704}"
MODEL_DIR="${MODEL_DIR:-/home/user/models/DeepSeek-V4-Flash-DSpark}"
NAME="${NAME:-ds4-dspark}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0,1}"
TP="${TP:-2}"
RESTART_POLICY="${RESTART_POLICY:-no}"

# Strongest TP2 DSpark decode row in the v9 sweep. Use BACKEND=b12x-a8 if
# optimizing prefill instead of decode.
BACKEND="${BACKEND:-lucifer-cutlass}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
GPU_MEM="${GPU_MEM:-0.93}"
GRAPH="${GRAPH:-128}"
DSPARK_TOKENS="${DSPARK_TOKENS:-5}"   # 0 = disable speculative decoding entirely
SAMPLE="${SAMPLE:-probabilistic}"
KV_DTYPE="${KV_DTYPE:-fp8}"           # fp8 (default) or auto (=model dtype, needs shorter MAX_MODEL_LEN)
REASONING_EFFORT="${REASONING_EFFORT:-high}"  # template only acts on 'max' ('high' = no-op prefix)
ENABLE_FLASHINFER_AUTOTUNE="${ENABLE_FLASHINFER_AUTOTUNE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

CACHE="${CACHE:-/home/user/llmstack/cache/$NAME}"
CONTAINER_TMP="${CONTAINER_TMP:-$CACHE/tmp}"

if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  echo "Missing model weights in $MODEL_DIR" >&2
  exit 1
fi

mkdir -p \
  "$CACHE/vllm" \
  "$CACHE/tilelang/tmp" \
  "$CACHE/tvm" \
  "$CACHE/triton" \
  "$CACHE/torchinductor" \
  "$CACHE/torch_extensions" \
  "$CACHE/flashinfer" \
  "$CONTAINER_TMP"

b12x_common_env=(
  -e VLLM_USE_B12X_WO_PROJECTION=1
  -e VLLM_USE_B12X_MHC=1
  -e VLLM_USE_B12X_MOE=1
  -e VLLM_USE_B12X_SPARSE_INDEXER=1
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x
  -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
  -e B12X_MLA_SM120_UNIFIED=1
  -e B12X_MHC_MAX_TOKENS=16384
  -e B12X_DENSE_SPLITK_TURBO=1
  -e B12X_W4A16_TC_DECODE=1
)

case "$BACKEND" in
  b12x-a8)
    BACKEND_ARGS=(--attention-backend B12X_MLA_SPARSE --moe-backend b12x --linear-backend b12x)
    BACKEND_ENV=(
      "${b12x_common_env[@]}"
      -e VLLM_USE_B12X_FP8_GEMM=1
      -e B12X_MOE_FORCE_A8=1
      -e B12X_MOE_FORCE_A16=0
    )
    ;;
  b12x-a16|b12x)
    BACKEND_ARGS=(--attention-backend B12X_MLA_SPARSE --moe-backend b12x --linear-backend b12x)
    BACKEND_ENV=(
      "${b12x_common_env[@]}"
      -e VLLM_USE_B12X_FP8_GEMM=1
      -e B12X_MOE_FORCE_A8=0
      -e B12X_MOE_FORCE_A16=1
    )
    ;;
  b12x-a8-dglin)
    BACKEND_ARGS=(--attention-backend B12X_MLA_SPARSE --moe-backend b12x)
    BACKEND_ENV=(
      "${b12x_common_env[@]}"
      -e VLLM_USE_B12X_FP8_GEMM=0
      -e B12X_MOE_FORCE_A8=1
      -e B12X_MOE_FORCE_A16=0
    )
    ;;
  lucifer-cutlass)
    BACKEND_ARGS=(--attention-backend FLASHINFER_MLA_SPARSE_DSV4 --kernel-config.moe_backend flashinfer_cutlass)
    BACKEND_ENV=(
      -e VLLM_ENABLE_PCIE_ALLREDUCE=1
      -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x
      -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
    )
    ;;
  lucifer-default)
    BACKEND_ARGS=(--attention-backend FLASHINFER_MLA_SPARSE_DSV4)
    BACKEND_ENV=(
      -e VLLM_ENABLE_PCIE_ALLREDUCE=1
      -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x
      -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
    )
    ;;
  *)
    echo "Unknown BACKEND=$BACKEND" >&2
    exit 2
    ;;
esac

SPEC_JSON=$(printf '{"model":"/model","method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"%s"}' "$DSPARK_TOKENS" "$SAMPLE")
SPEC_ARGS=(--speculative-config "$SPEC_JSON")
if [ "$DSPARK_TOKENS" = "0" ]; then
  SPEC_ARGS=()
fi

FLASHINFER_AUTOTUNE_FLAG=()
if [ "$ENABLE_FLASHINFER_AUTOTUNE" = "1" ]; then
  FLASHINFER_AUTOTUNE_FLAG=(--enable-flashinfer-autotune)
else
  FLASHINFER_AUTOTUNE_FLAG=(--no-enable-flashinfer-autotune)
fi

EAGER_FLAG=()
if [ "$ENFORCE_EAGER" = "1" ]; then
  EAGER_FLAG=(--enforce-eager)
fi

sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
sudo docker run -d \
  --name "$NAME" \
  --restart "$RESTART_POLICY" \
  --gpus all \
  --runtime nvidia \
  --ipc host \
  --shm-size 32g \
  --network host \
  --init \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v "$MODEL_DIR":/model:ro \
  -v "$CACHE":/cache:rw \
  -v "$CONTAINER_TMP":/container-tmp:rw \
  -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e VLLM_HOST_IP=127.0.0.1 \
  -e GLOO_SOCKET_IFNAME=lo \
  -e NCCL_SOCKET_IFNAME=lo \
  -e CUTE_DSL_ARCH=sm_120a \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_DSPARK_REPLICATE_MARKOV_W1=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e TMPDIR=/container-tmp \
  -e XDG_CACHE_HOME=/cache \
  -e VLLM_CACHE_DIR=/cache/vllm \
  -e TILELANG_CACHE_DIR=/cache/tilelang \
  -e TILELANG_TMP_DIR=/cache/tilelang/tmp \
  -e TVM_CACHE_DIR=/cache/tvm \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  "${BACKEND_ENV[@]}" \
  "$IMAGE" \
  /bin/bash -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; exec vllm serve "$@"' \
  -- /model \
  --served-model-name deepseek-v4-flash-dspark \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --kv-cache-dtype "$KV_DTYPE" \
  --block-size 256 \
  --load-format auto \
  "${EAGER_FLAG[@]}" \
  --tensor-parallel-size "$TP" \
  --disable-custom-all-reduce \
  --decode-context-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --max-cudagraph-capture-size "$GRAPH" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill \
  "${FLASHINFER_AUTOTUNE_FLAG[@]}" \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --enable-prompt-tokens-details \
  --enable-force-include-usage \
  --enable-request-id-headers \
  --default-chat-template-kwargs.thinking=true \
  --default-chat-template-kwargs.reasoning_effort="$REASONING_EFFORT" \
  "${SPEC_ARGS[@]}" \
  "${BACKEND_ARGS[@]}" \
  --enable-prefix-caching

echo "$NAME deepseek-v4-flash-dspark $BACKEND TP=$TP GPUS=$GPUS PORT=$PORT MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS GRAPH=$GRAPH FLASHINFER_AUTOTUNE=$ENABLE_FLASHINFER_AUTOTUNE ENFORCE_EAGER=$ENFORCE_EAGER"
