# AuctionNet Learning Guide

A guide to understanding how the bidding models work and how they are evaluated, grounded in this repo's code. Read it alongside the files it points to.

## 1. The one-sentence mental model

Every algorithm in this repo solves the same tiny decision: **once per time step, pick a single number `α` (a "bid coefficient"); then bid `α × pValue` on every ad opportunity that step.** Everything else — RL, LP, transformers — is just different machinery for choosing `α` well under a budget and a cost-per-acquisition (CPA) constraint.

Before any code: the whole problem is *"how aggressive should I bid right now, given how much budget and time I have left, and how the market is behaving?"*

## 2. The MDP (the spine — learn it first)

| Element | What it is | Where in code |
|---|---|---|
| **State** (16-dim) | time-left, budget-left, and rolling averages of bid / least-winning-cost / pValue / conversion / win-rate / volume | built in `train_data_generator.py` (offline) and rebuilt live in each strategy's `bidding()` |
| **Action** `α` | one scalar bid coefficient for the step | `action = total_bid / total_value` in the data |
| **Reward** | conversions won (`reward`) or summed pValue (`reward_continuous`) | `train_data_generator.py` |
| **Transition** | `(state, action, reward, next_state, done)` | one row of `training_data_all-rlData.csv` |

**Key realization:** the dataset is converted from *per-impression* rows into *per-(advertiser, time-step)* RL transitions (48,384 total). This is a textbook **offline RL** dataset — you learn a policy purely from logged data, never interacting with the live auction during training.

→ Read first: `strategy_train_env/bidding_train_env/train_data_generator/train_data_generator.py` (lines ~100–145 build state/action/reward). Then `bidding_train_env/strategy/base_bidding_strategy.py` for the interface every strategy implements.

## 3. How a trained model actually bids (worked example: IQL)

The most important file to internalize — `bidding_train_env/strategy/iql_bidding_strategy.py`:

```
bidding(timeStep, pValues, ..., history...):
    1. rebuild the SAME 16-dim state from live history   (lines 46–92)
    2. normalize it with the saved normalize_dict         (lines 97–98)
    3. alpha = self.model(state)                           (line 101)  ← the learned policy
    4. bids = alpha * pValues                              (line 103)
    return bids
```

**The single most important concept:** step 1 must reconstruct the *exact same features* the training data used (§2), or the model sees a different input distribution than it trained on. Compare `bidding()` lines 46–92 against `train_data_generator.py` lines 100–118 — that train/inference feature parity is where bid agents most often silently break.

## 4. The algorithm families (in learning order)

Start simple, then add one idea at a time. All offline-RL methods exist to solve **one** problem: the dataset only shows certain actions, so a naive learner overestimates the value of actions it never saw ("out-of-distribution" / distributional shift). Each method is a different fix.

**A. Heuristics / control — no learning**
- **Abid** — fixed `α` for the whole episode. The baseline floor.
- **PID** — a control loop that nudges `α` up/down to track a target spend/CPA. `simul_bidding_env/strategy/pid_bidding_strategy.py`. Understand "pacing" without any ML.

**B. Optimization**
- **OnlineLP** — treats each tick as a knapsack/LP: given predicted opportunity values and costs, solve for the bidding that maximizes value under budget. Reads the *raw* CSVs to fit, not the RL transitions. `run/run_onlinelp.py` + `baseline/onlineLp/`.

**C. Offline RL — the heart of the repo** (`baseline/<algo>/`)
- **BC (Behavior Cloning)** — pure supervised: regress `α` from state, imitating the data. No reward, no value function. The simplest learner; your reference point.
- **IQL** — learns a value function *without ever querying unseen actions*: a V-net via **expectile regression** (`calc_value_loss`), twin Q-nets (`calc_q_loss`), and an actor trained by **advantage-weighted regression** (`calc_policy_loss`: `exp(Q−V) × log π`). Read `baseline/iql/iql.py` end-to-end (~280 lines); the `step()` method shows the whole training tick.
- **CQL** — adds a term that *pushes down* Q-values on out-of-distribution actions (conservatism). Compare its loss to IQL's.
- **BCQ** — constrains the policy to stay near actions present in the data (generative action model). Note: slowest to train here (~71 min).
- **TD3+BC** — standard actor-critic (TD3) **plus** a behavior-cloning regularizer so the actor maximizes Q while staying close to the data. The "minimal change to online RL that makes it work offline."

**D. Sequence modeling**
- **Decision Transformer** — reframes control as sequence prediction: feed (return-to-go, state, action) tokens and autoregressively predict the next action. No value function. `baseline/dt/`. In our runs it overspent and got penalized — a good case study in how return-conditioning can misbehave.

→ Suggested path: **Abid → PID → BC → IQL → (CQL, BCQ, TD3+BC) → OnlineLP → DT.**

## 5. How evaluation works (two very different harnesses)

**Offline** (`run/run_evaluate.py`): one advertiser replays `period-7.csv`; your agent bids against the *recorded* `leastWinningCost`. Win if `bid ≥ leastWinningCost`; conversions are `binomial(pValue)`. Then:
```
score = penalty · reward,   penalty = (cpa_constraint / cpa)²  if cpa > constraint else 1
```
⚠️ As shipped it instantiates the agent at **budget=100, cpa=2**, so expected conversions <1 → scores ≈ 0. Low signal (see `docs/report.md`). Good for understanding the scoring formula, not for ranking.

**Online** (root `main_test.py` → `run/run_test.py`): the real thing — your agent competes against **47 other agents** in the full simulator. Read `simul_bidding_env/Controller/Controller.py` (builds the agent field + environment) and `BiddingEnv.simulate_ad_bidding` (the GSP auction: top bidders win slots, pay ~the next bid, win→expose→convert). The `adjust_over_cost` loop (drops bids when an agent would overspend) is worth tracing — it enforces budget constraints mid-auction.

The `analysis_info` per run (budget_used, cpa_exceedance, win_pv_ratio) is the diagnostic dashboard — it's how we explained *why* DT scored low (spent 100% of budget, blew the CPA constraint → heavy penalty).

## 6. Hands-on exercises (best way to learn)

1. **Trace one bid:** print `alpha` in `iql_bidding_strategy.bidding()` and run the offline eval — watch `α` change as budget depletes.
2. **Break feature parity:** zero out one state feature in `bidding()` and see the score move — proves §3's point.
3. **Compare losses:** read `calc_value_loss`/`calc_policy_loss`/`calc_q_loss` in `iql.py` and map each to the MDP in §2.
4. **Diff two algorithms:** open `baseline/iql/iql.py` vs `baseline/td3_bc/` and find the *one* idea that differs (the BC regularizer).
5. **Re-run with a tweak:** flip the `reward` → `reward_continuous` toggle in `run_iql.py`, retrain, re-eval, compare.

## 7. Glossary

- **pValue** — P(conversion | exposed); the predicted value of an opportunity.
- **α / bid coefficient** — the single action; scales pValue into a bid.
- **GSP** — Generalized Second Price auction; winner pays ~the next-highest bid.
- **leastWinningCost** — the price floor to win (4th-highest bid for 3 slots).
- **CPA** — cost per acquisition; the constraint that the penalty enforces.
- **Offline RL / distributional shift** — learning from fixed logs; the core difficulty every RL baseline here is engineered around.

## 8. Where to go next

- Results and observations from the actual training/eval runs: `docs/report.md`.
- The execution plan and commands: `docs/plan.md`.
- The original paper (background, environment design): `docs/AuctionNet_2412.10798.pdf`.
