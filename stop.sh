#!/usr/bin/env bash
# 兼容入口：停止 z100 vLLM 生产服务。实际操作在 start.sh（统一入口）。
#
#   bash stop.sh
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" stop
