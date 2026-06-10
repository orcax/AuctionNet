# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AuctionNet is a NeurIPS 2024 benchmark for **bid decision-making in large-scale ad auctions**. It has three loosely-coupled parts:

- `simul_bidding_env/` — the auction **simulator** (ad-opportunity generation, the auction mechanism, simulation control, and 48 competitor strategies). Self-contained; needs no external data to run.
- `strategy_train_env/` — **training framework** for baseline bidding algorithms (IQL, BC, BCQ, CQL, TD3+BC, OnlineLP, Decision-Transformer). Needs the downloaded dataset.
- `pre_generated_dataset/` — only download links for the ~80GB raw auction dataset (500M+ records); the data itself is not in the repo.

## Environment setup

The project targets **Python 3.9** (code uses `from collections import Iterable` in `run/run_test.py`, removed in 3.10+). README documents conda; this repo also runs under uv:

```bash
uv venv --python 3.9
uv pip install -r requirements.txt
```

Two known gaps in `requirements.txt` that surface on a fresh install:

- **`torch==1.12.0` fails to load on modern kernels** (`libtorch_cpu.so: cannot enable executable stack`). Install a newer CPU wheel instead: `uv pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu`.
- **`einops` is imported but not listed.** Install it: `uv pip install einops`.

## Common commands

All commands assume the venv (`uv run python …`, or activate `.venv`).

```bash
# Online evaluation — runs the simulator with your player strategy vs. 47 competitors.
# Self-contained: the default neuripsPvGen synthesizes traffic (seeded), no dataset needed.
uv run python main_test.py

# Training (run from inside strategy_train_env/, needs the downloaded dataset under data/traffic/):
cd strategy_train_env
python bidding_train_env/train_data_generator/train_data_generator.py   # raw csv -> trajectory data
python main/main_iql.py      # also: main_bc.py, main_bcq.py, main_cql.py, main_td3_bc.py,
                             #       main_onlineLp.py, main_decision_transformer.py
python main/main_test.py     # offline evaluation of a trained strategy
```

There is no lint or unit-test setup. `main_test.py` (root) is the integration smoke test.

## Architecture: the online evaluation loop

Entry `main_test.py` → `run/run_test.py:run_test()`. Everything is wired via **gin** config in `config/test.gin` (agent counts, traffic volume `PVNUM`, episodes/ticks, `GENERATE_LOG`, and `pv_generator_type`). Key objects:

- **`Controller`** (`simul_bidding_env/Controller/Controller.py`) builds the world: instantiates the PV generator, the `BiddingEnv`, and a fixed roster of **48 competitor agents** (`initialize_agents()` hard-codes the strategy mix per category). It also slots in the user's `player_agent` at `player_index`.
- **PV generator** — `NeurIPSPvGen` (default, synthesizes impression opportunities from seeded distributions + bundled `.pkl`/`.csv` stats under `PvGenerator/model_utils/data/`) or `ModelPvGen` (deep generative, needs `einops`/torch). Selected by `Controller.pv_generator_type` in the gin file.
- **`BiddingEnv`** (`Environment/BiddingEnv.py`) runs the auction: `simulate_ad_bidding()` returns winners, slots, costs, exposures, conversions, least-winning-cost.
- The loop in `run_test()` iterates episodes × ticks: every agent's `bidding()` produces bids, the env simulates the auction, then **`adjust_over_cost()` iteratively zeros out bids** for agents that would exceed their remaining budget and re-runs the auction until no one overspends.
- **`PlayerAnalysis`** / **`BiddingTracker`** (`Tracker/`) record per-tick metrics and (when `GENERATE_LOG=True`) write raw trajectory CSVs to `data/log/` — this is how new training datasets are produced.

## The bidding strategy interface

Every strategy (competitor and player) subclasses `BaseBiddingStrategy` (`strategy_train_env/bidding_train_env/strategy/base_bidding_strategy.py`, mirrored in `simul_bidding_env/strategy/`) and implements:

- `reset()` — restore `remaining_budget`.
- `bidding(timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid, historyAuctionResult, historyImpressionResult, historyLeastWinningCost)` — return a bid array for all opportunities in the period.

**Selecting the player strategy:** `run_test()` imports `PlayerBiddingStrategy` from `strategy_train_env/bidding_train_env/strategy/__init__.py`. Switch strategies by editing which line is uncommented there (default: `IqlBiddingStrategy`). If that import fails, the runner falls back to `PidBiddingStrategy`.

**Model loading gotcha:** the RL/generative player strategies `torch.jit.load` a trained model from `strategy_train_env/saved_model/<Strategy>test/` (gitignored, produced by training). Running online eval before training raises a "file does not exist" error. For a no-training smoke run, copy a bundled model from `simul_bidding_env/strategy/official_agent/<Strategy>test/` into the matching `saved_model/` path.

## Layout of the training framework (`strategy_train_env/`)

Each algorithm follows the same three-file pattern, mirroring the auction-env strategy interface:

- `bidding_train_env/baseline/<algo>/` — model definition.
- `main/main_<algo>.py` — thin entry point that calls `run/run_<algo>.py`.
- `bidding_train_env/strategy/<algo>_bidding_strategy.py` — the `BaseBiddingStrategy` wrapper used at evaluation time.

`train_data_generator.py` reads raw `data/traffic/period-*.csv` and writes per-period `*-rlData.csv` trajectory files used by training.
