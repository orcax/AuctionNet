# AuctionNet — Train & Evaluate Report

Date: 2026-06-10. Goal: follow **README.md** end-to-end via **uv** — process data, train all baselines, run offline + online evaluation. No paper reproduction.

## TL;DR

- Trained **all 7 trainable baselines** (IQL, BC, BCQ, CQL, TD3+BC, Decision Transformer, OnlineLP) on the full 21-period dataset and evaluated each with the repo's offline and online harnesses.
- **Online ranking (rank_score):** CQL > BCQ > OnlineLP > IQL ≈ BC > TD3+BC > DT.
- The repo's **offline eval is low-signal** (instantiates the player at budget=100/cpa=2 → expected conversions <1 → near-zero scores for everyone). The **online** 48-agent evaluation is the meaningful comparison.

---

## Environment

- Run via **uv**, not conda (per project convention). `.venv` = Python 3.9.21.
- **torch 2.2.2+cpu** (the pinned `torch==1.12.0` wheel fails to load on this kernel: `libtorch_cpu.so: cannot enable executable stack`). CPU-only.
- Added **`einops`** (imported by the PV-generator model utils but missing from `requirements.txt`).
- Dataset: 21 periods (`period-7.csv … period-27.csv`, 77 GB) downloaded from the OSS links in `pre_generated_dataset/readme_dataset.md`.

## Deviations from README (and why)

| Deviation | Reason |
|---|---|
| `uv` instead of conda | project convention |
| torch 2.2.2+cpu instead of 1.12.0 | pinned wheel won't load on this kernel |
| added `einops` | imported but missing from `requirements.txt` |
| `step_num` 100 → 20000 in `run_bcq.py`, `run_cql.py`, `run_td3_bc.py` | shipped value is a debug placeholder (CQL even has an `if i==8000` branch); bumped to match IQL/BC so the RL baselines are comparably trained |

---

## Pipeline & commands

All training / offline-eval from inside `strategy_train_env/`; online eval from repo root.

```bash
# 1. Data processing (once, ~28 min): raw 77 GB CSVs -> RL transitions
cd strategy_train_env
uv run python bidding_train_env/train_data_generator/train_data_generator.py
#   -> data/traffic/training_data_rlData_folder/training_data_all-rlData.csv  (29 MB, 48,384 transitions)

# 2. Train (per algorithm)
uv run python main/main_iql.py    # + main_bc / main_decision_transformer / main_onlineLp / main_bcq / main_cql / main_td3_bc
#   -> saved_model/<Algo>test/

# 3. Select player strategy: edit bidding_train_env/strategy/__init__.py (one active import)
# 4. Offline eval:  cd strategy_train_env && uv run python main/main_test.py
# 5. Online eval:   (repo root)            uv run python main_test.py
```

The eval sweep was automated by `strategy_train_env/data/eval_logs/run_eval.sh` (rewrites `__init__.py` per baseline, runs offline+online, restores the default).

---

## Training results

| Algorithm | Train steps | Train time | Model saved |
|---|---|---|---|
| IQL | 20000 | 3m18s | `saved_model/IQLtest/iql_model.pth` |
| BC | 20000 | ~1m55s | `saved_model/BCtest/bc_model.pth` |
| Decision Transformer | 10000 | ~6m31s | `saved_model/DTtest/dt.pt` |
| OnlineLP | LP fit (raw CSVs) | ~14m26s | `saved_model/onlineLpTest/period.csv` |
| BCQ | 20000 (bumped) | ~71m17s | `saved_model/BCQtest/bcq_model.pth` |
| CQL | 20000 (bumped) | ~11m39s | `saved_model/CQLtest/cql_model.pth` |
| TD3+BC | 20000 (bumped) | ~5m32s | `saved_model/TD3_bctest/td3_bc_model.pth` |

All exited 0. BCQ was by far the slowest (~71 min) — its per-step update is much heavier than the other RL baselines.

---

## Evaluation results

### Online (48-agent simulator, `config/test.gin` defaults: `neuripsPvGen`, 2 episodes, 48 ticks)

`rank_score` = sum of the penalized score over the two evaluated player slots (player_index 0 and 1). Higher is better.

| Rank | Algorithm | rank_score | reward [slot0, slot1] | budget_used | cpa_exceedance |
|---|---|---|---|---|---|
| 1 | **CQL** | **0.003187** | [42, 61] | [0.63, 0.84] | [-0.04, 0.76] |
| 2 | BCQ | 0.002895 | [33, 43] | [0.54, 0.48] | [0.10, 0.30] |
| 3 | OnlineLP | 0.002830 | [41, 36] | [0.58, 0.46] | [0.14, 0.37] |
| 4 | IQL | 0.002604 | [37, 44] | [0.52, 0.59] | [0.02, 0.65] |
| 5 | BC | 0.002601 | [35, 52] | [0.55, 0.71] | [0.18, 0.67] |
| 6 | TD3+BC | 0.002378 | [32, 44] | [0.52, 0.60] | [-0.06, 0.68] |
| 7 | DT | 0.001060 | [34, 43] | [1.00, 1.00] | [0.95, 1.89] |

### Offline (single advertiser on `period-7.csv`, player at budget=100 / cpa=2)

| Algorithm | Total Reward | Total Cost | Score |
|---|---|---|---|
| BC | 1.0 | 99.96 | 0.00040 |
| BCQ | 1.0 | 99.99 | 0.00040 |
| TD3+BC | 1.0 | 99.99 | 0.00040 |
| IQL | 0.0 | 99.92 | 0.0 |
| CQL | 0.0 | 99.97 | 0.0 |
| DT | 0.0 | 99.98 | 0.0 |
| OnlineLP | 0.0 | 0.0 | 0.0 |

---

## Observations

- **Offline eval is low-signal as shipped.** `run_evaluate.py` calls `PlayerBiddingStrategy()` with no args → `BaseBiddingStrategy` defaults **budget=100, cpa=2** (not the advertiser's real ~2900–7500 budget). Conversions are `binomial(pValue≈0.0006)`; at budget 100 the agent wins ~10³ impressions → **expected reward <1**, so results are noise (0 or 1) and Scores ≈ 0 regardless of algorithm. Not a training failure — treat online as the real signal. (OnlineLP bid below market everywhere at budget 100, so it won nothing → cost 0.)
- **DT is the clear online outlier (lowest).** Its `analysis_info` shows **budget_consumer_ratio = 1.0** (spends the entire budget) and **cpa_exceedance ≈ 0.95–1.89** (badly overshoots the CPA constraint), so the CPA penalty `(cpa_constraint/cpa)^2` crushes its score. It acquires conversions but far too expensively.
- **CQL / BCQ lead online** with disciplined spending (budget_used 0.5–0.8) and low CPA exceedance — they respect the constraint better, so less penalty.
- **Absolute scores are tiny** because `score = penalty · reward` for one player in one category; the ranking, not the magnitude, is what's informative.

## Issues hit & fixes

1. **torch 1.12.0 won't load** (`cannot enable executable stack`) → installed `torch==2.2.2+cpu`.
2. **`einops` missing** from requirements → installed.
3. **`unzip` not on the box** → extracted the dataset zips with Python `zipfile`.
4. **Flaky download (connection resets)** → resumable curl with `--retry-all-errors` + outer retry loop.
5. **BCQ/CQL/TD3+BC `step_num=100`** placeholder → bumped to 20000.
6. **Stdout buffering** made background logs look empty → monitored via per-run log files / process CPU instead.

## How to reproduce

1. `uv venv --python 3.9 && uv pip install -r requirements.txt`
2. `uv pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu && uv pip install einops`
3. `bash scripts/download_data.sh` (downloads + extracts the 21 periods)
4. `cd strategy_train_env && uv run python bidding_train_env/train_data_generator/train_data_generator.py`
5. Train: `uv run python main/main_<algo>.py` for each baseline.
6. Eval: `bash data/eval_logs/run_eval.sh` (or manually flip `strategy/__init__.py` and run offline/online).

Raw per-run logs: `strategy_train_env/data/eval_logs/eval_<algo>_{offline,online}.log` and `strategy_train_env/data/train_logs/<algo>.log`.
