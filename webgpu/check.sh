#!/bin/zsh
# 一次性跑一个 WebGPU 页面：起服务器 → 开浏览器 → 读结果 → 全关掉。
# 浏览器和服务器都活在这条命令里，是因为进程一旦交给后台，命令结束时会被回收。
PAGE=${1:-attn.html}
WAIT=${2:-10}
PORT=${3:-8791}
cd "$(dirname "$0")" || exit 1

python3 -m http.server "$PORT" --bind 127.0.0.1 >/tmp/pw_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
sleep 1

playwright-cli close-all >/dev/null 2>&1
playwright-cli open about:blank >/tmp/pw_open.log 2>&1
playwright-cli goto "http://127.0.0.1:$PORT/$PAGE?n=$(date +%s)" >/tmp/pw_goto.log 2>&1
sleep "$WAIT"
playwright-cli eval "document.getElementById('t').innerText + (document.getElementById('log').textContent ? '\n' + document.getElementById('log').textContent : '')" 2>&1 | sed -n '1,40p'
playwright-cli close-all >/dev/null 2>&1
