#!/usr/bin/env bash
# Evaluate each trained baseline: flip strategy/__init__.py to the algo,
# run offline (strategy_train_env/main/main_test.py) + online (repo-root main_test.py).
# Logs to strategy_train_env/data/eval_logs/eval_<algo>_{offline,online}.log.
# Restores the original __init__.py at the end.
# Usage:  bash scripts/run_eval.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ST_ENV="$REPO_ROOT/strategy_train_env"
PY="$REPO_ROOT/.venv/bin/python"
LOG="$ST_ENV/data/eval_logs"
INIT="$ST_ENV/bidding_train_env/strategy/__init__.py"
mkdir -p "$LOG"

# algo_key : "module ClassName"
declare -A IMP=(
  [iql]="iql_bidding_strategy IqlBiddingStrategy"
  [bc]="bc_bidding_strategy BcBiddingStrategy"
  [onlinelp]="onlinelp_bidding_strategy OnlineLpBiddingStrategy"
  [dt]="dt_bidding_strategy DtBiddingStrategy"
  [bcq]="bcq_bidding_strategy BcqBiddingStrategy"
  [cql]="cql_bidding_strategy CqlBiddingStrategy"
  [td3_bc]="td3_bc_bidding_strategy TD3_BCBiddingStrategy"
)

write_init() { printf 'from .%s import %s as PlayerBiddingStrategy\n' "$1" "$2" > "$INIT"; }

cp "$INIT" "$INIT.bak"   # preserve original (all commented options)

ORDER=(iql bc dt onlinelp bcq cql td3_bc)
for key in "${ORDER[@]}"; do
  read -r module cls <<< "${IMP[$key]}"
  echo ">>> Evaluating $key  ($cls)"
  write_init "$module" "$cls"

  ( cd "$ST_ENV" && PYTHONUNBUFFERED=1 "$PY" main/main_test.py ) > "$LOG/eval_${key}_offline.log" 2>&1
  echo "    offline exit=$? -> eval_${key}_offline.log"

  ( cd "$REPO_ROOT" && PYTHONUNBUFFERED=1 "$PY" main_test.py ) > "$LOG/eval_${key}_online.log" 2>&1
  echo "    online  exit=$? -> eval_${key}_online.log"
done

mv "$INIT.bak" "$INIT"   # restore original
echo ">>> Eval complete; __init__.py restored"
