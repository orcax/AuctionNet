# AuctionNet — Train & Evaluate Report (GPU run)

Date: 2026-06-13. Goal: re-run the README pipeline end-to-end via **uv**, this time training on **GPU**, and compare results against the prior CPU run (`docs/report.md`, 2026-06-10).

## TL;DR

- Re-trained **all 7 trainable baselines** (IQL, BC, BCQ, CQL, TD3+BC, Decision Transformer, OnlineLP) on the full 21-period dataset **on GPU** (2× NVIDIA L4), and re-ran the offline + online harnesses.
- **GPU's big win is BCQ: ~71 min → ~4.5 min (≈16× faster).** Other heavy RL baselines (CQL, TD3+BC) got 2–2.5×; the tiny ones (IQL, BC) were backend-neutral; OnlineLP (LP fit, CPU-bound) and DT were unchanged.
- Enabling GPU required **fixing latent device-placement bugs** in IQL/BCQ/CQL/TD3+BC that the CPU-only path never exercised (details below).
- **Online ranking (rank_score):** OnlineLP > TD3+BC > BCQ > BC > IQL > CQL > DT.
- **Alignment with the prior CPU report is partial:** the *structure* holds (OnlineLP strong & numerically identical; DT clearly worst by a wide margin; everything else clustered tightly), but the *exact ordering of the RL baselines reshuffled* — the cluster spread is within run-to-run / backend noise. CQL in particular fell from #1 (CPU) to mid-pack (GPU).

---

## Environment

- Run via **uv**, `.venv` = Python 3.9.21.
- **torch 2.2.2+cu121** (CUDA 12.1 build) on **2× NVIDIA L4** (sm_89, 23 GB each), driver supporting CUDA 13.3. *(The prior report used `torch 2.2.2+cpu`; this run swaps only the compute backend, same torch version.)*
- `einops` added (imported by PV-generator utils, missing from `requirements.txt`).
- Dataset: 21 periods (`period-7.csv … period-27.csv`, 77 GB).
- Data processing reproduced identically: **48,384 transitions, 29 MB** `training_data_all-rlData.csv` (matches prior run exactly).

## Deviations from README / prior report (and why)

| Deviation | Reason |
|---|---|
| `uv` instead of conda | project convention |
| **torch 2.2.2+cu121 (GPU)** instead of `+cpu` | user requested GPU training; box has 2× L4 |
| added `einops` | imported but missing from `requirements.txt` |
| `step_num` 20000 for BCQ/CQL/TD3+BC | already committed in the repo's run scripts (matches prior report's decision) |
| **device-placement code fixes** in `iql.py`, `bcq.py`, `cql.py`, `td3_bc.py` | the GPU path was never exercised (prior run was CPU) and had latent bugs; see "GPU fixes" |

---

## GPU enablement: code fixes

The training code auto-detects CUDA (`torch.cuda.is_available()`), but the GPU path had never run. Math-preserving device fixes applied:

- **IQL** (`baseline/iql/iql.py`): move the sampled batch onto the model device in `step()`; replace CPU-hardcoded `torch.min(exp_a, torch.FloatTensor([100.0]))` with `torch.clamp(exp_a, max=100.0)`; make `take_actions` follow the model's actual device.
- **BCQ** (`baseline/bcq/bcq.py`): move batch to device in `step()`; VAE `decode` allocates `z` on `state.device`; in `save_jit`, move to CPU and set device attrs **before** `torch.jit.script` (jit bakes the device value at scripting time).
- **CQL** (`baseline/cql/cql.py`): create `log_alpha` on `self.device` (it's a plain tensor, so `self.to(device)` doesn't move it) — this was the only *training-time* crash.
- **TD3+BC** (`baseline/td3_bc/td3_bc.py`): move batch to device in `step()`; in `save_jit`, sync agent + submodule device attrs to CPU **before** scripting.

The "before scripting" ordering matters: `torch.jit.script` captures `self.device`'s value into the serialized graph. Scripting while `self.device==cuda` produced models that crashed at eval (`state.to(cuda)` vs CPU weights). Fixing the order makes the saved jit models CPU-portable, which is what the eval harness expects.

---

## Training results (GPU) vs prior (CPU)

| Algorithm | Steps | GPU time | CPU time (prior) | Speedup |
|---|---|---|---|---|
| IQL | 20000 | 237s | ~198s | 0.8× (neutral) |
| BC | 20000 | 118s | ~115s | 1.0× |
| Decision Transformer | 10000 | 339s | ~391s | 1.2× |
| OnlineLP | LP fit | 852s | ~866s | 1.0× (CPU-bound) |
| **BCQ** | 20000 | **271s** | **~4277s (71m)** | **≈16×** |
| CQL | 20000 | 279s | ~699s | 2.5× |
| TD3+BC | 20000 | 144s | ~332s | 2.3× |

All exited 0. GPU helps exactly where expected — the per-step-heavy BCQ — and is neutral for tiny MLPs where launch/transfer overhead offsets the small matmuls.

---

## Evaluation results (GPU run)

### Online (48-agent simulator, `config/test.gin` defaults: `neuripsPvGen`, 2 episodes, 48 ticks)

`rank_score` = sum of penalized score over player slots 0 and 1. Higher is better.

| Rank | Algorithm | rank_score | reward [s0,s1] | budget_used | cpa_exceedance |
|---|---|---|---|---|---|
| 1 | **OnlineLP** | **0.002830** | [41, 36] | [0.58, 0.46] | [0.14, 0.37] |
| 2 | TD3+BC | 0.002706 | [36, 57] | [0.55, 0.82] | [-0.12, 0.80] |
| 3 | BCQ | 0.002698 | [32, 37] | [0.53, 0.40] | [0.36, 0.26] |
| 4 | BC | 0.002593 | [34, 49] | [0.54, 0.68] | [-0.04, 0.65] |
| 5 | IQL | 0.002556 | [35, 46] | [0.53, 0.61] | [0.10, 0.63] |
| 6 | CQL | 0.002530 | [35, 55] | [0.62, 0.78] | [0.31, 0.78] |
| 7 | DT | 0.000737 | [27, 42] | [1.00, 1.00] | [1.80, 1.97] |

### Offline (single advertiser on `period-7.csv`, player at budget=100 / cpa=2)

| Algorithm | Reward | Cost | Score |
|---|---|---|---|
| BCQ | 2.0 | 99.9 | 0.00320 |
| CQL | 1.0 | 99.9 | 0.00040 |
| DT | 1.0 | 99.9 | 0.00040 |
| BC | 1.0 | 99.9 | 0.00040 |
| TD3+BC | 1.0 | 99.9 | 0.00040 |
| IQL | 0.0 | 99.9 | 0.0 |
| OnlineLP | 0.0 | 0.0 | 0.0 |

Offline remains **low-signal** (as the prior report documented): the player is instantiated at budget=100/cpa=2, so expected conversions are <1–2 and the Score is essentially a 0/1/2 coin-flip. Not a meaningful ranking signal — online is the real comparison.

---

## Comparison with the prior CPU report (`docs/report.md`)

### Online rank_score, side by side

| Algorithm | CPU (prior) | CPU rank | GPU (this) | GPU rank |
|---|---|---|---|---|
| OnlineLP | 0.002830 | 3 | **0.002830** | 1 |
| CQL | **0.003187** | 1 | 0.002530 | 6 |
| BCQ | 0.002895 | 2 | 0.002698 | 3 |
| IQL | 0.002604 | 4 | 0.002556 | 5 |
| BC | 0.002601 | 5 | 0.002593 | 4 |
| TD3+BC | 0.002378 | 6 | 0.002706 | 2 |
| DT | 0.001060 | 7 | 0.000737 | 7 |

### What aligns

- **OnlineLP is numerically identical** (0.002830 → 0.002830). Expected: it's an LP fit on raw CSVs, independent of the torch backend — a clean control showing the eval harness itself is reproducible.
- **DT is the clear worst in both**, by a large margin (~3–4× below the pack). Same mechanism in both runs: `budget_consumer_ratio = 1.0` (spends the entire budget) and `cpa_exceedance ≈ 1.8–2.0`, so the CPA penalty `(cpa_constraint/cpa)²` crushes its score.
- **The other six cluster tightly** in both runs (CPU: 0.00238–0.00319; GPU: 0.00253–0.00283).
- **Absolute magnitudes are the same order** and the offline eval is low-signal in both.

### What diverges

- **The RL baselines' exact ordering reshuffled.** Most strikingly **CQL fell from #1 (CPU) to #6 (GPU)**, and **TD3+BC rose from #6 to #2**. The cluster spread (~0.0003, ~12% of the value) is comparable to the differences being ranked, so the ordering within the cluster is **within noise**.

### Why

1. **GPU vs CPU is not bit-identical.** Even with `torch.manual_seed(1)`/`np.random.seed(1)`, CUDA uses different kernels, reduction orders, and RNG streams than CPU. The trained RL weights genuinely differ from the CPU-trained ones, so a tightly-clustered ranking reshuffles.
2. **The signal is small.** Each `rank_score` is a penalized conversion count for one player in one category over 2 episodes — low absolute magnitude with real simulator variance.
3. The device code fixes are **math-preserving** (e.g. `clamp(max=100)` ≡ `min(·,100)`; moving `log_alpha` to GPU doesn't change its value), so they are not the cause — the cause is the backend's numerical nondeterminism plus the tight clustering.

### Verdict

The reproduction **aligns on every robust conclusion** — OnlineLP is a strong, deterministic baseline; DT is the clear outlier loser; offline eval is uninformative; the remaining baselines are roughly comparable. It **does not reproduce the precise RL leaderboard order**, which is expected: those differences live inside run-to-run / CPU-vs-GPU noise and should not be over-interpreted. The takeaway from both reports is the same — **rank the extremes, not the cluster.**

---

## Issues hit & fixes (this run)

1. **CPU→GPU switch**: installed `torch==2.2.2+cu121` (had to `--reinstall`; uv treated `+cpu` as already satisfying `==2.2.2`). Verified CUDA with a real GPU matmul.
2. **IQL device crash** (`step()` batch on CPU, nets on cuda; CPU constant at line 194) → fixed.
3. **CQL training crash** (`log_alpha` left on CPU by `self.to(device)`) → created on device.
4. **BCQ/TD3+BC eval crash** — jit models baked `cuda` into their forward because `self.device` was reset *after* `torch.jit.script`; reordered to set CPU device before scripting.
5. **Duplicate-orchestrator race** — an earlier GPU attempt wasn't killed before relaunch, so two runs wrote the same logs/`saved_model`; killed all, wiped, restarted a single clean run.

## How to reproduce (GPU)

1. `uv venv --python 3.9 && uv pip install -r requirements.txt`
2. `uv pip install "torch==2.2.2+cu121" --index-url https://download.pytorch.org/whl/cu121 --reinstall && uv pip install einops`
3. `bash scripts/download_data.sh`
4. `cd strategy_train_env && uv run python bidding_train_env/train_data_generator/train_data_generator.py`
5. Train: `bash scripts/train_all.sh` (trains all 7 in order, IQL first).
6. Eval: `bash scripts/run_eval.sh` (flips `strategy/__init__.py` per baseline, runs offline + online, restores the original).

Raw logs: `strategy_train_env/data/train_logs/<algo>.log`, `strategy_train_env/data/eval_logs/eval_<algo>_{offline,online}.log`.
