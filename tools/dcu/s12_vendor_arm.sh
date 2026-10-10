#!/usr/bin/env bash
# S12: fair baseline -- run the *production* stack (image vendor vLLM 0.18.1+das
# plus the production phaseA + int8_sdot4 injection layers) on the current
# fabric state, so "production 122.47 ms/token" can be replaced by a
# same-fabric number.
#
#   tools/dcu/s12_vendor_arm.sh <NAME> <TP> <PORT> <MAXLEN> <DEVICES> [EXTRA...]
#
# Runs in the no-memory-limit shadow container `z100-s12` (the production
# container `qwen38-vllm` is off-limits and must not be started/exec'd).
# Mirrors /home/lyh/server/serve_27b_tp4.sh defaults: ENFORCE_EAGER=1,
# GPU_UTIL=0.85, PREFIX=1, MBT=2048, CUSTOM_AR=0.
#
# Host artifacts -> $WS/.tmp-12/vendor-<NAME>.log
set -u
NAME=$1; TP=$2; PORT=$3; MAXLEN=$4; DEVICES=$5; shift 5
EXTRA="$*"
WS=/home/lyh/z100-vllm
C="${S12_VENDOR_CT:-z100-s12}"
LOGDIR="$WS/.tmp-12"
mkdir -p "$LOGDIR"
ELOG="$LOGDIR/vendor-$NAME.log"
: > "$ELOG"

docker exec "$C" bash -c "mkdir -p /mnt/game/s12" || exit 1

# Inner script is written to a file first: quoting a vllm command line inside
# `docker exec -d "..."` eats the quotes (documented in serve_27b_tp4.sh).
docker exec -i "$C" bash -c "cat > /tmp/_s12_vendor_inner.sh" <<EOF
#!/usr/bin/env bash
for f in /opt/dtk/env.sh /etc/profile.d/env.sh; do [ -f "\$f" ] && source "\$f" >/dev/null 2>&1; done
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:\${LD_LIBRARY_PATH}
export TRITON_CACHE_DIR=/mnt/game/triton-cache
# Copied verbatim from /home/lyh/server/serve_27b_tp4.sh -- without
# TRITON_HIP_LLD_PATH the stock image triton fails with
# "RuntimeError: PassManager::run failed" (measured 2026-10-09).
export TRITON_HIP_LLD_PATH=/opt/dtk/llvm/bin/ld.lld
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800 VLLM_RPC_TIMEOUT=10800000
export VLLM_ROCM_USE_SKINNY_GEMM=0 TORCH_BLAS_PREFER_HIPBLASLT=0 PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES=$DEVICES
export INT8_SDOT4_DISABLE=0 INT8_SDOT4_MAX_M=8 INT8_SDOT4_GEMM=1 INT8_SDOT4_GEMM_CFG=0 INT8_SDOT4_NT=64
export VLLM_ROCM_CUSTOM_AR=0
echo \$\$ > /tmp/s12_vendor_${NAME}.pid
cd /tmp && exec vllm serve /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 \\
  --tensor-parallel-size $TP --pipeline-parallel-size 1 \\
  --max-model-len $MAXLEN --max-num-batched-tokens 2048 \\
  --gpu-memory-utilization 0.85 \\
  --enforce-eager \\
  --skip-mm-profiling \\
  --enable-prefix-caching --mamba-cache-mode align \\
  --served-model-name qwen3.8-27b \\
  --host 127.0.0.1 --port $PORT $EXTRA
EOF
docker exec "$C" bash -c "sed -i 's/\r\$//' /tmp/_s12_vendor_inner.sh"
nohup timeout 3600 docker exec "$C" bash -c "bash /tmp/_s12_vendor_inner.sh > /tmp/serve-${NAME}.log 2>&1" >/dev/null 2>&1 &

t0=$(date +%s)
READY=0
while true; do
  if docker exec "$C" bash -c "grep -aq 'Application startup complete' /tmp/serve-${NAME}.log" 2>/dev/null; then READY=1; break; fi
  # Only real startup failures.  A bare 'Error' also matches the benign
  # "ModuleNotFoundError: No module named 'gdn_decode_build'" warning line,
  # which made the first attempt kill a healthy engine (2026-10-09).
  if docker exec "$C" bash -c "tail -120 /tmp/serve-${NAME}.log | grep -aE 'EngineCore failed to start|Engine core initialization failed|ValueError: |RuntimeError: |raise ValueError' | grep -av 'ModuleNotFoundError' | grep -aq ." 2>/dev/null; then
    echo "[vendor] FATAL marker"; break
  fi
  if [ $(( $(date +%s) - t0 )) -gt "${READY_TIMEOUT:-1800}" ]; then echo "[vendor] TIMEOUT"; break; fi
  sleep 10
done
docker exec "$C" bash -c "cat /tmp/serve-${NAME}.log" > "$ELOG" 2>&1
echo "[vendor] ready=$READY after $(( $(date +%s) - t0 ))s (log $ELOG)"

if [ "$READY" = "1" ]; then
  docker exec "$C" bash -c "grep -aE 'Selected|Available KV|Maximum concurrency|ACTIVE|Gfx906|TorchSdot4|TritonInt8' /tmp/serve-${NAME}.log | tail -12" | tee -a "$ELOG"
  for mode in decode ttft conc8; do
    echo "[vendor] bench $mode $(date -Is)"
    docker exec "$C" bash -c "
      for f in /opt/dtk/env.sh /etc/profile.d/env.sh; do [ -f \"\$f\" ] && source \"\$f\" >/dev/null 2>&1; done
      export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:\${LD_LIBRARY_PATH}
      cd /tmp
      case $mode in
        decode) EX='--random-input-len 128 --random-output-len 128 --num-prompts 2 --num-warmups 1 --max-concurrency 1' ;;
        ttft)   EX='--random-input-len 3800 --random-output-len 1 --num-prompts 1 --num-warmups 0 --max-concurrency 1' ;;
        conc8)  EX='--random-input-len 512 --random-output-len 128 --num-prompts 16 --num-warmups 2 --max-concurrency 8' ;;
      esac
      vllm bench serve --backend openai-chat --base-url http://127.0.0.1:$PORT \
        --endpoint /v1/chat/completions --model /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 \
        --served-model-name qwen3.8-27b --dataset-name random --seed 0 \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --temperature 0 --ignore-eos \$EX
    " 2>&1 | tee "$LOGDIR/vendor-$NAME-bench-$mode.log"
  done
fi

docker exec "$C" bash -c "kill \$(cat /tmp/s12_vendor_${NAME}.pid) 2>/dev/null; pkill VLLM 2>/dev/null" || true
sleep 20
docker exec "$C" bash -c "pkill -9 VLLM 2>/dev/null" || true
echo "[vendor] done name=$NAME ready=$READY"
