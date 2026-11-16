#!/usr/bin/env bash
# z100 vLLM 生产服务 —— 统一运维入口（唯一的"怎么启动"答案）。
#
#   bash start.sh              启动 + 等 /health + 校验 spec/policy（默认）
#   bash start.sh --no-wait    只触发 systemd，立即返回
#   bash start.sh stop         停止（端口空出，清理孤儿 worker）
#   bash start.sh restart      重启 + 等 /health
#   bash start.sh status       单元 / 端口 / 健康 / .so / 指纹 一览
#   bash start.sh health       只查一次 /health
#   bash start.sh logs [-f]    最近 200 行日志（-f 跟随）
#   bash start.sh config       干跑：打印解析后的 env + 命令行（不启动）
#
# 配置真源：serve_deploy.sh（env）+ launch_mtp2.sh（CLI）。
# systemd 单元只负责 ExecStart 拉起 serve_deploy.sh；两者配置已对拍一致。
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=z100-vllm-mtp2
PORT="${PORT:-8000}"
SUDO=""
[ "$(id -u)" != "0" ] && SUDO="sudo -n"

say() { echo "[$UNIT] $*"; }
die() { echo "[$UNIT] ERROR: $*" >&2; exit 1; }

main_pid() { systemctl show "$UNIT" -p MainPID --value 2>/dev/null; }

listener_pid() {
  ss -tlnpH "sport = :${PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1
}

http_health() {
  curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:${PORT}/health" 2>/dev/null || true
}

wait_health() {
  say "waiting for /health (cold start with compile cache ~4-8 min)..."
  local pid c lp i=0
  pid=$(main_pid)
  for i in $(seq 1 180); do
    sleep 5
    c=$(http_health)
    lp=$(listener_pid)
    # 只认单元主进程持有的监听：刚 start 时仍在排空的旧服务也会回 200。
    # 某些受限环境（无 /proc 权限）ss -p 读不到 pid，此时退化为单元 active
    # 且 MainPID 存活，并在下面明确告警。
    if [ -z "$lp" ] && [ "$c" = "200" ] && [ -n "$pid" ] && [ "$pid" != "0" ] \
       && systemctl is-active --quiet "$UNIT" && kill -0 "$pid" 2>/dev/null; then
      lp=$pid
      lp_unverified=1
    fi
    [ "$c" = "200" ] && [ -n "$pid" ] && [ "$pid" != "0" ] && [ "$lp" = "$pid" ] && break
  done
  if [ "$c" != "200" ] || [ "$lp" != "$pid" ]; then
    echo "[$UNIT] ERROR: not healthy (http=$c listener=${lp:-none} expected=${pid:-none}) after ~$((i * 5))s"
    echo "[$UNIT] last journal lines:"
    $SUDO journalctl -u "$UNIT" --since "-15 min" --no-pager 2>/dev/null | tail -25
    exit 1
  fi
  say "health 200 after ~$((i * 5))s (listener pid $lp = unit main pid)"
  [ "${lp_unverified:-0}" = "1" ] && \
    say "WARN: ss could not report the listener owner here; verified via unit state + MainPID only"
  say "--- spec line (expect num_speculative_tokens: 2) ---"
  $SUDO journalctl -u "$UNIT" --since "-30 min" --no-pager 2>/dev/null \
    | grep "spec=" | tail -1 || echo "NO SPEC LINE"
  say "--- policy fingerprint ---"
  $SUDO journalctl -u "$UNIT" --since "-30 min" --no-pager 2>/dev/null \
    | grep "policy: FP " | tail -1 || echo "NO POLICY LINE"
  say "done"
}

cmd_start() {
  say "starting $UNIT ..."
  $SUDO systemctl start "$UNIT" || die "systemctl start failed"
  if [ "${1:-}" = "--no-wait" ]; then
    say "unit kicked (--no-wait); follow with: bash start.sh logs -f"
    return 0
  fi
  wait_health
}

cmd_stop() {
  say "stopping $UNIT ..."
  $SUDO systemctl stop "$UNIT" || true
  local i
  for i in $(seq 1 30); do
    pgrep -f "entrypoints[.]cli" >/dev/null || break
    sleep 2
  done
  pkill -9 -f 'VLLM[:]:Worker' 2>/dev/null || true
  sleep 3
  if pgrep -f "entrypoints[.]cli" >/dev/null; then
    say "WARNING: a vllm server process is still running:"
    pgrep -af "entrypoints[.]cli"
    return 1
  fi
  say "stopped; port ${PORT} is free, no vllm server running"
}

cmd_status() {
  echo "unit      : $UNIT = $(systemctl is-active "$UNIT" 2>/dev/null) / $(systemctl is-enabled "$UNIT" 2>/dev/null)"
  echo "main pid  : $(main_pid)"
  echo "listener  : ${PORT} -> pid $(listener_pid || true)"
  echo "health    : $(http_health)"
  echo "rocm .so  : $(readlink -f "$REPO/vllm/_rocm_C.abi3.so" 2>/dev/null || echo MISSING)"
  echo "md5       : $(md5sum "$REPO/vllm/_rocm_C.abi3.so" 2>/dev/null | cut -d' ' -f1 || echo n/a)"
  echo "env file  : $REPO/serve_deploy.sh"
  $SUDO journalctl -u "$UNIT" --since "-7 days" --no-pager 2>/dev/null \
    | grep "policy: FP " | tail -1 || true
}

case "${1:-start}" in
  start)   cmd_start "${2:-}" ;;
  --no-wait) cmd_start --no-wait ;;
  stop)    cmd_stop ;;
  restart)
    say "restarting $UNIT ..."
    $SUDO systemctl restart "$UNIT" || die "systemctl restart failed"
    wait_health
    ;;
  status)  cmd_status ;;
  health)
    c=$(http_health)
    echo "http=$c"
    [ "$c" = "200" ] || exit 1
    ;;
  logs)
    if [ "${2:-}" = "-f" ]; then
      $SUDO journalctl -u "$UNIT" -n 200 -f
    else
      $SUDO journalctl -u "$UNIT" -n 200 --no-pager
    fi
    ;;
  config)
    cd "$REPO" && PRINT_ONLY=1 bash serve_deploy.sh
    ;;
  -h|--help|help)
    sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
  *)
    die "unknown command '${1}' (try: start|stop|restart|status|health|logs|config)"
    ;;
esac
