#!/usr/bin/env bash
# DD-FP R7 GPU server runner -- thin launcher for r7/r7run.py.  No sudo anywhere.
# Deliberately NOT using `set -e` / `set -u` / `pipefail`: a failing step must never abort the shell before its log is saved.
#   bash run7.sh preflight --server-tree <your DD-FP tree> | --no-server-tree      (read-only)
#   bash run7.sh prepare | e2 | e1 | e6 | e3 | e4 --confirm E4 | status | collect | verify-pkg | selftest
# On a non-zero exit the terminal stays open (press Enter) when run interactively; set R7_NO_PAUSE=1 to disable (e.g. under tmux/nohup).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -z "$HERE" ] || [ ! -f "$HERE/r7/r7run.py" ]; then echo "ERROR: cannot locate r7/r7run.py next to run7.sh" >&2; exit 2; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
PY="${R7_ORCH_PYTHON:-}"
if [ -z "$PY" ]; then for cand in python3 python; do if command -v "$cand" >/dev/null 2>&1; then PY="$(command -v "$cand")"; break; fi; done; fi
if [ -z "$PY" ]; then echo "ERROR: python3 not found in PATH" >&2; exit 2; fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then echo "ERROR: $PY is older than Python 3.8" >&2; exit 2; fi
if [ "$#" -eq 0 ]; then echo "usage: bash run7.sh <preflight|prepare|setup-venv|e2|e1|e6|e3|e4|status|collect|verify-results|verify-pkg|selftest> [options]   (see README_KO.md)"; exit 2; fi

"$PY" -B -I "$HERE/r7/r7run.py" "$@"
rc=$?
if [ "$rc" -ne 0 ] && [ -t 0 ] && [ -t 1 ] && [ -z "${R7_NO_PAUSE:-}" ]; then
  echo; echo "[run7.sh] exit code $rc. Logs and partial results are kept. Press Enter to close this terminal."; read -r _
fi
exit "$rc"
