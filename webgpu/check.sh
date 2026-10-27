#!/bin/zsh
# 一次性跑一个 WebGPU 页面：起服务器 → 打开 → 读结果 → 关掉。
# 不常驻，是因为上次那个后台服务器被中断了，而留一个长期进程也不是这个脚本要做的事。
PAGE=${1:-attn.html}
WAIT=${2:-10}
PORT=${3:-8791}
cd "$(dirname "$0")" || exit 1
python3 -m http.server "$PORT" --bind 127.0.0.1 >/tmp/pw_server.log 2>&1 &
SRV=$!
sleep 1
playwright-cli open about:blank >/dev/null 2>&1 || true
playwright-cli goto "http://127.0.0.1:$PORT/$PAGE?n=$(date +%s)" >/tmp/pw_goto.log 2>&1
sleep "$WAIT"
playwright-cli eval "document.getElementById('t').innerText + (document.getElementById('log').textContent ? '\n' + document.getElementById('log').textContent : '')" 2>&1 | sed -n '1,40p'
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
