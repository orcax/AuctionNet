#!/usr/bin/env bash
# Train all 7 baselines in plan order (IQL first validates the loop).
# Uses the repo .venv. Logs + timings per algo to strategy_train_env/data/train_logs/.
# Usage:  bash scripts/train_all.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ST_ENV="$REPO_ROOT/strategy_train_env"
PY="$REPO_ROOT/.venv/bin/python"
LOG="$ST_ENV/data/train_logs"
mkdir -p "$LOG"
SUMMARY="$LOG/train_summary.txt"
: > "$SUMMARY"

# algo_key : main script
ALGOS=(
  "iql:main_iql.py"
  "bc:main_bc.py"
  "dt:main_decision_transformer.py"
  "onlinelp:main_onlineLp.py"
  "bcq:main_bcq.py"
  "cql:main_cql.py"
  "td3_bc:main_td3_bc.py"
)

for entry in "${ALGOS[@]}"; do
  key="${entry%%:*}"; script="${entry##*:}"
  echo ">>> Training $key ($script)"
  start=$(date +%s)
  ( cd "$ST_ENV" && PYTHONUNBUFFERED=1 "$PY" "main/$script" ) > "$LOG/$key.log" 2>&1
  rc=$?
  dur=$(( $(date +%s) - start ))
  echo "$key exit=$rc time=${dur}s" | tee -a "$SUMMARY"
  [ "$rc" -ne 0 ] && echo "!!! $key FAILED (exit $rc) — see $LOG/$key.log" | tee -a "$SUMMARY"
done
echo ">>> Training complete" | tee -a "$SUMMARY"
