# AuctionNet — Train & Evaluate (following README.md)

## Goal

Follow the **README.md** workflow end-to-end, run through **uv** (not conda): process the data, **train the baseline strategy(ies)**, then run both the **offline** and **online** evaluations the README describes. No paper reproduction, no custom Figure-10 harness, no comparison targets — just make the documented pipeline work and report what it outputs.

**Scope (decided): all baselines** — IQL, BC, BCQ, CQL, TD3+BC, OnlineLP, Decision Transformer (plus PID/Abid reference points which need no training). IQL runs first to validate the loop, then the rest.

**Decisions locked:**
- **Train steps:** bump `step_num` 100→**20000** in `run/run_bcq.py`, `run/run_cql.py`, `run/run_td3_bc.py` so all RL baselines are comparably trained (documented deviation from shipped default).
- **Evaluation:** run **both** offline (`main/main_test.py`, Score on period-7) and online (root `main_test.py`, `rank_score`) for each baseline.

---

## README workflow → exact uv commands

All training/offline-eval commands run from inside `strategy_train_env/`; the online eval runs from the repo root. Use `uv run python ...`.

### 0. Environment (done)
- `.venv` (Python 3.9), torch 2.2.2+cpu (pinned 1.12.0 won't load here), `einops` added.
- Dataset present: 21 periods (`period-7.csv … period-27.csv`, 77 GB) in `strategy_train_env/data/traffic/`.

### 1. Data processing (README "Data Processing") — once, shared
```bash
cd strategy_train_env
uv run python bidding_train_env/train_data_generator/train_data_generator.py
```
→ writes `data/traffic/training_data_rlData_folder/training_data_all-rlData.csv` (+ per-period files).

### 2. Strategy training (README "Strategy Training")
```bash
cd strategy_train_env
uv run python main/main_iql.py        # README's example
# optional others:
uv run python main/main_bc.py
uv run python main/main_decision_transformer.py
uv run python main/main_onlineLp.py
uv run python main/main_bcq.py
uv run python main/main_cql.py
uv run python main/main_td3_bc.py
```
→ saves model + `normalize_dict.pkl` to `saved_model/<Algo>test/`.

### 3. Select the player strategy (README snippet)
Edit `strategy_train_env/bidding_train_env/strategy/__init__.py` so exactly the trained algorithm is the active import, e.g.:
```python
from .iql_bidding_strategy import IqlBiddingStrategy as PlayerBiddingStrategy
```

### 4. Offline evaluation (README "Offline Evaluation")
```bash
cd strategy_train_env
uv run python main/main_test.py
```
→ logs Reward / Cost / CPA-real / CPA-constraint / **Score** for the player on `period-7.csv`.

### 5. Online evaluation (README "Online Evaluation")
```bash
# from repo root; hyperparameters in config/test.gin (left at defaults)
uv run python main_test.py
```
→ prints `rank_score` + `analysis_info` for the player in the 48-agent simulator.
Config stays as shipped: `config/test.gin` (default `pv_generator_type = "neuripsPvGen"`, `NUM_EPISODE=2`, `NUM_TICK=48`).

---

## Per-algorithm facts (verified from `run/run_*.py`)

| Algorithm | Strategy class (for step 3) | Train input | Default steps | Save dir |
|---|---|---|---|---|
| IQL | `IqlBiddingStrategy` | combined rlData | 20000 | `saved_model/IQLtest` |
| BC | `BcBiddingStrategy` | combined rlData | 20000 | `saved_model/BCtest` |
| BCQ | `BcqBiddingStrategy` | combined rlData | **100 ⚠** | `saved_model/BCQtest` |
| CQL | `CqlBiddingStrategy` | combined rlData | **100 ⚠** | `saved_model/CQLtest` |
| TD3+BC | `TD3_BCBiddingStrategy` | combined rlData | **100 ⚠** | `saved_model/TD3_bctest` |
| Decision Transformer | `DtBiddingStrategy` | combined rlData | 10000 | `saved_model/DTtest` |
| OnlineLP | `OnlineLpBiddingStrategy` | **raw `data/traffic/`** | LP fit | `saved_model/onlineLpTest` |
| PID / Abid | (in `simul_bidding_env`) | none | — | — |

---

## Execution order

1. **Phase 1** — data generation (background; slow, parses 77 GB).
2. **Phase 2** — train **IQL** (validates the whole loop).
3. **Phase 3** — set IQL as player → **offline eval** → **online eval**; record outputs.
4. **Phase 4 (optional)** — repeat steps 2–3 for the other baselines.
5. **Phase 5 — Report.** Once the above are done, write `docs/report.md` summarizing the findings (see Deliverable).

---

## Risks / notes

- **torch 2.2.2 vs pinned 1.12.0** — numbers won't match the authors' exactly; not a goal here.
- **BCQ/CQL/TD3+BC `step_num=100`** — a debug placeholder; at defaults they barely train. Following the README literally runs them as-is; flag the near-untrained result, or bump to ~20000 if we want meaningful models (decision when we get there).
- **Phase 1 memory** — the generator concatenates all per-period frames before writing; watch RSS for OOM on 77 GB.
- **OnlineLP / DT differ** — OnlineLP reads raw CSVs (not rlData); DT saves via `save_net` (others `save_jit`). Handled by their own scripts.
- **Disk** — `data/zips/` (~77 GB) can be deleted now that extraction succeeded.

---

## Deliverable

For each trained baseline: a trained model in `saved_model/`, plus its **offline Score** and **online `rank_score`/`analysis_info`** captured in a simple results table. No paper comparison.

**Final report → `docs/report.md`** (written in Phase 5, after the runs complete). It should cover:
- What was run: environment (uv, Python 3.9, torch 2.2.2+cpu), commands, which baselines, config used.
- Results table: per baseline → offline Score (Reward/Cost/CPA) and online `rank_score` + `analysis_info`.
- Observations: training behavior (loss/steps), runtimes, anything notable (e.g. `step_num=100` placeholders, OnlineLP/DT differences).
- Deviations from the README and why (uv instead of conda, torch version, einops, any `step_num` changes).
- Issues hit and how resolved; reproduction instructions to re-run.
