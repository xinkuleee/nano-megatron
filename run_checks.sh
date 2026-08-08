#!/usr/bin/env bash
# Run every check. The socket-free ones work anywhere; the distributed matrix
# needs an environment that permits binding sockets (a normal shell, not a
# restricted sandbox).
set -uo pipefail

cd "$(dirname "$0")"

echo "== sharding checks (no sockets) =="
python3 -m tests.check_sharding || exit 1

echo
echo "== tensor-parallel math (no sockets) =="
python3 -m tests.check_tp_math || exit 1

echo
echo "== logical gradient clipping (no sockets) =="
python3 -m tests.check_grad_clip || exit 1

echo
echo "== distributed correctness matrix =="
if python3 -c "
import socket, sys
try:
    s = socket.socket(); s.bind(('127.0.0.1', 0)); s.close()
except OSError:
    sys.exit(1)
" 2>/dev/null; then
    python3 -m tests.check_parallel --matrix || exit 1
else
    echo "  FAIL: this environment does not permit binding sockets, which"
    echo "  gloo requires. The distributed correctness matrix was not run."
    exit 1
fi
