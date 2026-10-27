#!/usr/bin/env bash
# S20 c=1 profiling arm: the production s19 FP8 + DFlash2 k=7 recipe plus a
# torch profiler with a bounded schedule.  Identical to the production launcher
# except for --profiler-config, so the traced numbers stay comparable.
set -u
export S19_MAXLEN=262144
export S19_LOGLEVEL=INFO
export S19_DRAFT_DTYPE=bfloat16
export S19_DRAFT_KV_DTYPE=bfloat16
export S19_LOG="${S19_LOG:-/mnt/game/dflash/serve-arm-profc1s.log}"
export VLLM_DCU_BF16=1
export VLLM_QWEN35_MTP_GDN_TRITON=1
export VLLM_GFX906_GDN_CHUNKED_PREFILL=1
PROF_DIR=${PROF_DIR:-/mnt/game/prof_c1s}
WAIT=${WAIT:-12}
WARMUP=${WARMUP:-3}
ACTIVE=${ACTIVE:-25}
rm -rf "$PROF_DIR"; mkdir -p "$PROF_DIR"
exec bash /src/tools/dcu/s19_dflash_arm.sh \
  fp8 4 8000 /mnt/game/models/Qwen3.8-27B-DFlash2-ms 7 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROF_DIR\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"torch_profiler_with_memory\":false,\"torch_profiler_with_flops\":false,\"wait_iterations\":$WAIT,\"warmup_iterations\":$WARMUP,\"active_iterations\":$ACTIVE}" \
  --override-generation-config '{"temperature":0.7,"repetition_penalty":1.05}' \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
  --host 0.0.0.0
