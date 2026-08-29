#!/usr/bin/env bash
# Run every check. The socket-free ones work anywhere; the distributed matrix
# needs an environment that permits binding sockets (a normal shell, not a
# restricted sandbox).
set -euo pipefail

cd "$(dirname "$0")"

if [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
else
    PYTHON=python3
fi

echo "== sharding checks (no sockets) =="
$PYTHON -m tests.check_sharding

echo
echo "== tensor-parallel math (no sockets) =="
$PYTHON -m tests.check_tp_math

echo
echo "== logical gradient clipping (no sockets) =="
$PYTHON -m tests.check_grad_clip

echo
echo "== data, checkpoint, and recomputation checks =="
$PYTHON -m pytest -q -m "not distributed"

echo
echo "== distributed correctness matrix =="
if $PYTHON -c "
import socket, sys
try:
    s = socket.socket(); s.bind(('127.0.0.1', 0)); s.close()
except OSError:
    sys.exit(1)
" 2>/dev/null; then
    $PYTHON -m tests.check_sequence_parallel
    $PYTHON -m tests.check_parallel --matrix
else
    echo "  FAIL: this environment does not permit binding sockets, which"
    echo "  gloo requires. The distributed correctness matrix was not run."
    exit 1
fi
