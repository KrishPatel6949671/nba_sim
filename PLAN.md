# nba-sim — Implementation Plan

This is the contractor-grade spec for building the NBA box score simulator. Each section lists files touched, concrete deliverables, and how we'll know it works.

> **Conventions used in this document**
> - File paths are relative to the repo root.
> - Tensor shapes are written `[B, P, D]` where `B`=batch, `P`=players-per-team (padded to 15), `D`=feature dim.
> - "Ships when" = the test(s) that must pass and the artifact(s) that must land on disk for the item to be considered done.

---

## 1. Problem statement and success criteria

### Problem

Given `(home_team, away_team, game_date, rosters_optional)`, produce a realistic full box score:

- Per team: final score, pace (possessions), offensive and defensive rating.
- Per player (rostered + active): MIN, PTS, FGM, FGA, 3PM, 3PA, FTM, FTA, OREB, DREB, REB, AST, STL, BLK, TOV, PF, +/-. (17 stats. We model 15 natively; REB is derived as OREB+DREB and +/- is derived from the team-level scoring margin allocated by on-court time.)

### What "production-usable" means concretely

A run counts as production-usable when, on the held-out season validation set, the v1 model satisfies all of:

| Criterion | Target | Measurement |
| --- | --- | --- |
| Constraint violation rate | **0.00%** on 10,000 simulated games | `tests/test_constraints.py` run over sampler output |
| Per-stat MAE vs. Poisson GLM baseline | **≥ 15% reduction** on PTS, REB, AST; ≥ 5% on others | `nba_sim.training.metrics.per_stat_mae` |
| Per-stat MAE vs. "predict player season average to date" | beats it on every counting stat | same |
| Calibration (predicted distribution coverage) | 80% predictive interval contains actual ≥ 78% of the time per stat | calibration bins |
| Team total MAE | PTS ≤ 8.5 per team, pace ≤ 3.5 poss | team-level rollup |
| Inference latency | 1 simulated game ≤ 50 ms on GPU, ≤ 500 ms on CPU, batched 128 games ≤ 1 s on GPU | `pytest -m slow` perf check |
| Reproducibility | Same seed → identical box score | determinism test |

Non-goals for v1 (explicit in §11): play-by-play, injuries mid-game, foul-outs, ejections, OT distribution, garbage-time dynamics.

---

## 2. Data pipeline

### 2.1 Sources

**Primary: `nba_api`.** Endpoints used (all accessed through thin wrappers in `src/nba_sim/data/fetch.py`):

| Endpoint | Purpose | Granularity |
| --- | --- | --- |
| `leaguegamefinder.LeagueGameFinder` | canonical list of games by season (incl. game_id, date, matchup, home/away) | per game |
| `boxscoretraditionalv3.BoxScoreTraditionalV3` | minutes, points, FG/3P/FT makes+attempts, rebound split, AST, STL, BLK, TOV, PF, +/- | per player, per game |
| `boxscoreadvancedv3.BoxScoreAdvancedV3` | offensive/defensive rating, pace, usage, TS%, eFG% — used as training targets and features | per team and per player, per game |
| `commonteamroster.CommonTeamRoster` | roster for a team on a given season | per team, per season |
| `playergamelog.PlayerGameLog` | redundancy check + unlocks older seasons with missing per-game flags | per player, per season |
| `commonplayerinfo.CommonPlayerInfo` | height, weight, position, draft year, country, experience | per player, lifetime |
| `leaguedashteamstats.LeagueDashTeamStats` | season-level team strength priors for cold-start and calibration | per team, per season |
| `scoreboardv2.ScoreboardV2` | live game context (rest days derived from prior-game dates) | per date |

**Fallback: Basketball Reference scraper** (`src/nba_sim/data/scrape_bref.py`). Used only when `nba_api` returns empty or partial data for seasons before ~2000, or for historical height/weight fields that `commonplayerinfo` is flaky on. Conservative: 3-second inter-request delay, respect robots.txt, cache aggressively.

### 2.2 Rate limiting and caching

- All `nba_api` calls go through `fetch.cached_call(endpoint, params)` which:
  - Hashes `(endpoint, params)` → cache key.
  - Reads from `data/raw/.cache/{endpoint}/{hash}.parquet` (for tabular) or `.json` (for the rare nested payload).
  - On miss, calls the endpoint with `tenacity` retries (exponential backoff, max 5 attempts, jitter), enforcing `NBA_API_MIN_DELAY_SECONDS` between requests via a module-level token bucket.
  - Writes to cache on success.
- Cache invalidation is **manual** via `nba-sim fetch --refresh <endpoint>`; historical data is immutable so this is rare. Current-season endpoints are re-fetched when `--season current` is passed.

### 2.3 Schemas

Defined as Pydantic v2 models in `src/nba_sim/data/schema.py`. Three layers:

**Raw layer** (`RawGame`, `RawPlayerBoxLine`, `RawTeamBoxLine`, `RawRoster`, `RawPlayerInfo`): mirror the API fields 1:1, all optional, string-typed where the API is messy. Goal: faithful record so we can re-derive interim without re-fetching.

**Interim layer** (`Game`, `PlayerBoxLine`, `TeamBoxLine`, `Roster`): typed and validated. `MIN` is parsed from `"mm:ss"` to float minutes. Shot splits verified (FGM ≤ FGA, etc.). DNP/inactive flags set. IDs are integers. `season` is the start year (2023–24 → 2023). Written to `data/interim/<season>/*.parquet`.

**Processed layer** (`TrainingRow`, `TeamTrainingRow`): denormalized, feature-engineered, ready for the DataLoader. One row per game-player pair with joined team context, matchup context, rolling features, and targets. Written to `data/processed/{train,val,test}.parquet`.

### 2.4 ETL

`src/nba_sim/data/etl.py` defines three functions, each a pure transform with a deterministic output path:

- `raw_to_interim(season: int) -> Path` — reads `data/raw/`, emits `data/interim/{season}/`.
- `interim_to_processed(splits: SplitSpec) -> Path` — reads all seasons in `splits.train ∪ val ∪ test`, joins rosters + rolling features (see §3), emits `data/processed/{train,val,test}.parquet`.
- `build_feature_tables(season: int) -> None` — precomputes rolling aggregates that multiple features share.

All ETL is **idempotent** (writes to a tmp file then renames) and **incremental** (skips a season whose output parquet already exists and is newer than all its inputs — mtime-based).

### 2.5 Edge cases

| Case | Handling |
| --- | --- |
| Traded players mid-season | One `PlayerBoxLine` row per game; team membership is *per game*, not per season. Rolling features are computed on the player regardless of team. |
| Two-way contracts | Flagged in roster; treated as regular players for modeling. A `two_way` feature is passed to the encoder. |
| DNPs / coach's decision | Row exists with `MIN=0` and `active=True`. Minutes head must be able to allocate 0 to rostered-but-not-played players. |
| Inactive list | Row exists with `MIN=0` and `active=False`. Excluded from the player set fed to the model — the model only sees the active roster for that game. |
| Games called early / short games | Team MIN total ≠ 240 flagged; dropped from training (`dropped=True` in interim) but counted in a QA report. |
| Historical data gaps (pre-1996) | v1 uses 2000+ only. Older seasons are out of scope. |
| Missing +/- pre-1997 | +/- is a *derived* output in v1 (from team margin × minute share), not a modeled stat, so pre-1997 games are fine. |

### 2.6 Ships when

- `pytest tests/test_data_schema.py -q` passes.
- `python scripts/fetch_all_seasons.py --seasons 2022,2023 && nba-sim build-features --seasons 2022,2023` produces `data/processed/{train,val,test}.parquet` with >0 rows and all schema validations pass.
- A sample of 100 games cross-checked against Basketball Reference box scores has < 0.1% cell-level disagreement.

---

## 3. Feature engineering

All features are computed from **prior games only** — we use rows with `game_date < target_game_date`. This is enforced in `features/rolling.py` by filtering on `date` before aggregation.

### 3.1 Feature list

**Player features** (`features/rolling.py` unless noted):

| Feature | Computation | Window |
| --- | --- | --- |
| `p_min_avg_{5,10,20}` | mean minutes | last 5 / 10 / 20 games played |
| `p_usage_avg_{5,10,20}` | mean USG% | same |
| `p_ts_{5,10,20}` | mean true shooting % | same |
| `p_pts_per_min_{5,10,20}` | PTS / MIN | same |
| `p_{stat}_per_min_{10}` | per-minute rate for FGA, 3PA, FTA, REB, AST, STL, BLK, TOV, PF | last 10 |
| `p_games_played_season` | count of games before target in current season | cumulative |
| `p_days_since_last_game` | date diff in days, capped at 30 | scalar |
| `p_b2b` | bool: played yesterday | scalar |
| `p_3in4`, `p_4in6` | schedule density flags | scalar |
| `p_season_role` | `{'starter', 'rotation', 'end-of-bench', 'inactive'}` based on trailing-10 MIN bucket | categorical |
| `p_age` | years since birthdate on game date | scalar |
| `p_experience` | NBA seasons as of game date | scalar |
| `p_height_in`, `p_weight_lbs` | from `commonplayerinfo` | scalar |
| `p_position` | 1-hot {PG, SG, SF, PF, C}, mapped from `POSITION` | categorical |
| `p_embedding_id` | integer ID into a learned player embedding table; cold-start handled in §7 | int |

**Team features** (`features/rolling.py`):

| Feature | Computation | Window |
| --- | --- | --- |
| `t_pace_{5,10}` | mean pace | last 5/10 |
| `t_off_rtg_{5,10}` | mean offensive rating | same |
| `t_def_rtg_{5,10}` | mean defensive rating | same |
| `t_win_pct_{10}` | wins / 10 | last 10 |
| `t_pts_avg_{10}`, `t_pts_allowed_{10}` | rolling | last 10 |
| `t_embedding_id` | int into team embedding table | int |

**Context features** (`features/context.py`):

| Feature | Computation |
| --- | --- |
| `is_home` | bool |
| `rest_days` | days since previous game, clipped 0..5 |
| `b2b` | bool |
| `season_phase` | `{'early', 'mid', 'late', 'playoffs'}` based on date within season |
| `day_of_week` | 0..6 |
| `month` | 1..12 |
| `altitude_ft` | arena altitude lookup (Denver, Utah are outliers) |
| `travel_miles_prev` | great-circle from previous game's arena |

**Matchup features** (`features/matchup.py`):

| Feature | Computation |
| --- | --- |
| `opp_def_rtg_{10}` | opponent's rolling DefRtg |
| `opp_pace_{10}` | opponent's rolling pace |
| `opp_def_rtg_vs_pos_{season}` | season DefRtg allowed to position `p_position` — important for rebounding stats |
| `h2h_last_meeting_margin` | point margin in the two teams' most recent meeting, or 0 |

### 3.2 Leakage risks and mitigation

- **Same-game leakage:** all rolling aggregates strictly use `date < target_date`. Unit-tested in `tests/test_features.py` with a property test: for any row, no contributing game has a date ≥ the target date.
- **Season continuity leakage:** splits are by **full season** (§4 below), so a player's form from the validation season never appears in training features.
- **Player embedding leakage:** player embeddings are learned from training seasons only; when a player appears in a validation season, the embedding is used **frozen**. New players get the cold-start path (§7).
- **Opponent rolling DefRtg leakage:** uses opponent's last 10 games *before* the target date, not the target game itself.

### 3.3 Ships when

- `pytest tests/test_features.py -q` passes, including a hypothesis property test that no rolling feature uses future data.
- For a known player-game (e.g., LeBron 2023-11-15), the feature row matches hand-computed values within 1e-6 for all numeric features.

---

## 4. Model architecture

### 4.1 Overview

A **two-stage hierarchical model**: team-level pace and efficiency first, then player-level allocation conditioned on team-level predictions.

```
         ┌──────────────────────────────┐
         │  Game context encoder        │  (home/away, rest, matchup, date)
         │  ctx: [B, D_ctx]             │
         └───────────────┬──────────────┘
                         │
      ┌──────────────────┼──────────────────┐
      │                  │                  │
      ▼                  ▼                  ▼
  Home roster       Away roster       Matchup features
  [B, P, D_p]       [B, P, D_p]       [B, D_m]
      │                  │
  Player encoder     Player encoder         │
      │                  │                  │
      ▼                  ▼                  │
  Attn-pooled        Attn-pooled            │
  home team rep      away team rep          │
  [B, D_t]           [B, D_t]               │
      └──────────┬───────┴───────┬──────────┘
                 │               │
                 ▼               ▼
       ┌─────────────────────────────────┐
       │  Team-level head                │
       │  → pace, off_rtg_home,          │
       │     off_rtg_away                │
       │  (3 scalars per team-pair)      │
       └────────────┬────────────────────┘
                    │  teacher-forced during training,
                    │  sampled during simulation
                    ▼
       ┌─────────────────────────────────┐
       │  Player allocation head          │
       │  Input: per-player encoding      │
       │         + team-level predictions │
       │  Outputs per player:             │
       │   - minutes (Dirichlet × 240)    │
       │   - usage share (Dirichlet)      │
       │   - per-minute rate params for   │
       │     each counting stat           │
       │   - shooting % params (Beta)     │
       └─────────────────────────────────┘
```

### 4.2 Tensor shapes

`B` = batch size, `P = 15` (padded max active roster), `D_p = 64` (player feature dim), `D_t = 128` (team rep dim), `D_ctx = 32` (context), `D_m = 16` (matchup).

| Module | Input | Output |
| --- | --- | --- |
| `PlayerEncoder` | `[B, P, D_p_raw]` + `[B, P]` embedding ids | `[B, P, D_t]` |
| `RosterAttentionPool` | `[B, P, D_t]` + mask `[B, P]` | `[B, D_t]` |
| `GameContextEncoder` | numerical+categorical ctx | `[B, D_ctx]` |
| `TeamHead` | `concat(home_t, away_t, ctx, matchup)` → `[B, 2*D_t + D_ctx + D_m]` | pace `[B]`, off_rtg `[B, 2]` |
| `PlayerAllocHead` | `[B, P, D_t]` per side + team preds | distribution params per player |

### 4.3 Distributions chosen per stat and why

All instantiated from `torch.distributions` so `log_prob` is used directly in the loss and `sample`/`rsample` is used at simulation time.

| Stat | Distribution | Rationale |
| --- | --- | --- |
| Minutes allocation per team | `Dirichlet(α) × 240` | Naturally sums to 240. α > 0 from softplus; α scale encodes "peakedness" (blowout benches give flatter α). Zeroed-out players use a mask that sets α to a tiny constant. |
| Team pace (possessions) | `Normal(μ, σ)`, then rounded post-hoc for display | Pace is well-approximated by Gaussian; σ is small (~3–4 poss). |
| Team off rating | `Normal(μ, σ)` | Same reason. |
| Field goal attempts per player | `NegativeBinomial(total_count, probs)` conditioned on player minutes | Count data, over-dispersed relative to Poisson. Total-count parameterization via `logits` keeps grads stable. |
| 3P attempts, FT attempts | `NegativeBinomial` | Same. |
| FG made given FGA | `Binomial(total_count=FGA, probs=p_fg)` | Bounded by FGA — makes ≤ attempts by construction. |
| 3PM given 3PA, FTM given FTA | `Binomial` | Same. |
| OREB, DREB, AST, STL, BLK, TOV, PF | `NegativeBinomial` | Count data, over-dispersed. |
| Minutes actually *played* by a player (for output) | Deterministic: `min_share × 240` rounded to nearest 0.1 | Sum-to-240 preserved by rounding-with-residual redistribution. |
| +/- | Derived: `team_margin × (player_min / avg_team_min_on_floor)` with a noise term | Not modeled as a primary head; too noisy and fully determined by team margin + who's on floor, which we don't model. |

### 4.4 How each constraint is enforced by construction

| Constraint | Mechanism |
| --- | --- |
| `sum(minutes) == 240` per team | Dirichlet over 15 slots, multiply by 240, then round-with-residual in the sampler. |
| `FGM ≤ FGA` (and for 3P, FT) | `Binomial(total_count=FGA_sampled, probs=sigmoid(...))`. |
| counts ≥ 0 and integer | all count heads are discrete distributions on non-negative integers. |
| `team_total_{stat} == sum(player_{stat})` | The only team-level *predicted* quantities are pace and efficiency; PTS/FGA/etc. totals are literally `sum` of sampled player values. No separate team totals predicted → no inconsistency possible. |
| minutes for DNP rostered players = 0 | Dirichlet mass for DNP slots is driven to ~0 by a learned gate `p_plays_in_this_game` (Bernoulli conditioned on player features); if 0, slot's α is set to 1e-4. |

### 4.5 Parameter count target

| Component | Params |
| --- | --- |
| Player embedding (5,000 players × 32) | 160,000 |
| Team embedding (40 × 16) | 640 |
| Player encoder MLP (2 layers, 64→128→128) | ~25,000 |
| Roster attention | ~50,000 |
| Context encoder | ~5,000 |
| Team head | ~30,000 |
| Player alloc head (distribution params) | ~200,000 |
| **Total target** | **~500,000** |

Small by modern standards; the dataset is ~700k player-game rows, so ~1.5 parameters per training example is reasonable and leaves room for regularization (§5).

### 4.6 Files

- `src/nba_sim/models/encoders.py` — `PlayerEncoder`, `RosterAttentionPool`, `GameContextEncoder`.
- `src/nba_sim/models/heads.py` — `TeamHead`, `PlayerAllocHead`, `DirichletMinutes`, `NegBinomialCountHead`, `BinomialPercentHead`.
- `src/nba_sim/models/hierarchical.py` — `HierarchicalBoxScoreModel` that composes the above and exposes `forward(batch) -> BoxScoreDistribution`.
- `src/nba_sim/models/losses.py` — `composite_nll(pred_distributions, targets, weights)`.
- `src/nba_sim/models/baseline_glm.py` — sklearn-based Poisson GLM + ridge regression for % stats.

### 4.7 Ships when

- `pytest tests/test_model_shapes.py -q` passes with forward-pass shape assertions for every module.
- `pytest tests/test_constraints.py -q` passes: 1,000 samples from a randomly-init'd model all satisfy every constraint.

---

## 5. Training

### 5.1 Loss

Composite negative log-likelihood, summed over all heads with weights tuned to balance gradient magnitudes:

```
L = λ_min * NLL_minutes_Dirichlet
  + λ_pace * NLL_pace
  + λ_rtg * NLL_off_rtg
  + Σ_stat λ_stat * NLL_stat_over_players
  + λ_gate * BCE_plays_in_this_game
```

Initial weights `λ_* = 1.0` then tuned once with a short grid search on the validation set (§6). A small L2 penalty on learned embeddings (partial pooling proxy, §5.4).

### 5.2 Optimizer and schedule

- Optimizer: `AdamW`, lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.95).
- Scheduler: `CosineAnnealingLR` over `max_epochs` with a 500-step linear warmup.
- Batch size: 64 games (~960 player-rows/batch). Fits in 8 GB VRAM comfortably.
- Epochs: 30, with early stopping on composite validation NLL (patience=5).
- Mixed precision: `torch.amp.autocast(dtype=bfloat16)` on CUDA, fp32 master weights, `GradScaler` disabled for bf16.
- Gradient clipping: `clip_grad_norm_(..., max_norm=1.0)`.
- Checkpointing: every epoch to `models/ckpt_epoch_{n}.pt`, plus `models/best.pt` tracked by val NLL.

### 5.3 Curriculum

Two-stage:

1. **Team-level pretraining (5 epochs).** Freeze player heads. Train only pace + off-rating heads with team-level targets. This stabilizes the shared team encoder before the noisier player heads activate.
2. **Joint training (25 epochs).** Unfreeze everything; train the full composite loss.

### 5.4 Regularization and partial pooling

Low-sample players (rookies, deep bench) would overfit if each got a fully free embedding. Partial pooling via:

- Each player embedding `e_i` is decomposed as `e_i = role_embedding[role(i)] + δ_i` with `δ_i` initialized to 0 and penalized by `λ_pool * ||δ_i||²` weighted inversely by career games (new players get stronger shrinkage).
- Dropout on the roster attention pool output (`p=0.1`).
- Label smoothing is NOT used — distributions are already well-calibrated via NB dispersion parameter.

### 5.5 Experiment tracking

**W&B** (offline by default; set `WANDB_MODE=online` to sync). Rationale: easy sweeps, artifact-lineage tracking, per-run config snapshots, and integrates with CLI with minimal boilerplate. We still write TensorBoard scalars as a fallback so users without W&B can diagnose.

Logged per step: composite loss, per-head NLL, grad-norm, lr, throughput. Per epoch: §6 metrics on train and val.

### 5.6 Files

- `src/nba_sim/training/dataset.py` — `BoxScoreDataset` reading parquet via Polars, collating to padded tensors with masks.
- `src/nba_sim/training/loop.py` — `train(config)` entry point called from the CLI.
- `src/nba_sim/training/metrics.py` — per-stat MAE, calibration bins, constraint-violation counter.
- `configs/train.yaml` — all hyperparameters and paths.

### 5.7 Ships when

- `nba-sim train --config configs/train.yaml --model hierarchical --max-epochs 2` runs end-to-end on a single season subset and produces `models/best.pt`, with loss strictly decreasing.
- Deterministic: same seed → identical loss curve to 1e-6.
- `python scripts/train.py --config configs/train.yaml` on the full dataset converges; best val composite NLL logged to W&B.

---

## 6. Evaluation

### 6.1 Validation protocol

**Season-held-out.** Splits:

- **Train:** seasons 2000–2021 (22 seasons).
- **Validation:** season 2022 (picks hyperparameters).
- **Test:** season 2023 (reported metrics, used once at the end).

This prevents roster-continuity leakage — a player's 2023 form never contributes to the 2022 model selection.

### 6.2 Metrics

Computed in `src/nba_sim/training/metrics.py` and logged per epoch / per evaluation.

| Metric | Definition | Reported per |
| --- | --- | --- |
| Per-stat MAE | `mean(|y - E[ŷ]|)` | stat, overall |
| Per-stat RMSE | same, squared | stat |
| Calibration (coverage of 80% PI) | `mean(y in [ŷ_10, ŷ_90])` | stat |
| Reliability-diagram bucket error | bucketed into 10 bins of predicted mean | stat |
| Team PTS MAE | per team per game | overall |
| Pace MAE | per team per game | overall |
| Constraint-violation rate | fraction of sampled games with any constraint broken | should be 0 |

### 6.3 Baselines

All implemented in `src/nba_sim/models/baseline_glm.py` and trained + evaluated with the same splits:

1. **"Season-average" baseline.** For a player on game date `d`, predict their average for each stat over all prior games in the current season. If no prior games, use the prior-season average; if rookie with no prior, use the positional mean.
2. **Poisson GLM.** For each counting stat, a Poisson GLM on the full feature vector. Percentages via Beta regression. Minutes via softmax across the roster with a secondary regression on total active players. Constraint-enforcement post-hoc by projection (for the baseline only; the NN doesn't need this).

The NN must beat both. If it doesn't, v1 is not shippable.

### 6.4 Diagnostics required for sign-off

- Reliability plot per stat (predicted-mean bucket vs. observed mean) saved to `reports/reliability_<stat>.png`.
- Per-team PTS scatter (pred vs. actual) saved to `reports/team_pts_scatter.png`.
- Residual analysis: per-stat residuals split by `season_phase`, `is_home`, `b2b` to detect systematic error.
- A "box score card" for 20 randomly picked test-set games: side-by-side predicted box vs. actual, rendered to markdown under `reports/samples/`.

### 6.5 Ships when

- `nba-sim evaluate --split test --checkpoint models/best.pt` writes the full metrics JSON and all diagnostic plots to `reports/`.
- Every success criterion from §1 is met on the *test* split.

---

## 7. Simulation API

### 7.1 Signature

```python
# src/nba_sim/simulate/api.py

def simulate_game(
    home_team: str,                      # 3-letter team abbr, e.g. "BOS"
    away_team: str,
    date: str | datetime.date,           # ISO "YYYY-MM-DD" or date
    home_roster: list[str] | None = None,  # player names or IDs; None = use that team's most recent active roster
    away_roster: list[str] | None = None,
    n_samples: int = 1,
    seed: int | None = None,
    device: str = "cuda",                # "cuda" or "cpu"
    checkpoint: str | Path = "models/best.pt",
    return_distributions: bool = False,  # if True, return raw distribution params alongside samples
) -> BoxScore | BoxScoreEnsemble:
    ...
```

- `n_samples == 1` → returns a `BoxScore` (single sample).
- `n_samples > 1` → returns a `BoxScoreEnsemble` with per-cell mean + 80% interval + all raw samples.
- `seed` is passed into `torch.Generator` for full determinism.

### 7.2 BoxScore types

Defined in `src/nba_sim/data/schema.py`:

```python
class PlayerBoxLine:
    player_id: int
    player_name: str
    min: float
    pts: int
    fgm: int; fga: int
    tpm: int; tpa: int
    ftm: int; fta: int
    oreb: int; dreb: int; reb: int
    ast: int; stl: int; blk: int; tov: int; pf: int
    plus_minus: float

class TeamBoxLine:
    team: str
    players: list[PlayerBoxLine]
    pts: int; pace: float; off_rtg: float; def_rtg: float

class BoxScore:
    home: TeamBoxLine
    away: TeamBoxLine
    date: datetime.date
```

### 7.3 Unseen-player handling (cold start)

Three-tier fallback, in `simulate/sampler.py::_resolve_player_embedding`:

1. **Player ID seen in training.** Use the trained embedding directly.
2. **Player ID unseen but we have `commonplayerinfo`** (rookie or new signing). Look up height/weight/position/draft_year/experience; run them through a small "player attribute → embedding" projection trained as an auxiliary head (§5.1 — a lightweight auxiliary loss reconstructs each seen player's embedding from its attributes). Use the projected embedding.
3. **No info at all.** Use the positional mean embedding `role_embedding[predicted_role]` with the role predicted from whatever attributes we have (default: `'end-of-bench'`).

This is tested in `tests/test_simulate_api.py` by injecting synthetic unseen IDs.

### 7.4 Sampling strategy

- **Single sample (`n_samples=1`)**: call `.sample()` on each distribution. Fast, representative.
- **Ensemble (`n_samples>1`)**: vectorize sampling by expanding the batch dim; all `n_samples` share one forward pass through the encoder.
- **"Mean" alternative**: `simulate_game(..., mode="mean")` returns the expected value of each distribution (`.mean`). Useful for reporting; not a valid box score because expected values of counts are rarely integers and don't sum to exactly 240 unless we project.

### 7.5 Determinism

- `simulate_game(seed=42)` → bit-exact same output on the same device.
- CUDA determinism flags set in `utils/seed.py::set_deterministic()`. GPU reproducibility comes at ~5–10% throughput cost — acceptable for single-game inference.

### 7.6 Ships when

- `pytest tests/test_simulate_api.py -q` passes, including:
  - determinism test,
  - unseen-player test,
  - constraint test on 100 samples,
  - shape/schema test.

---

## 8. CLI and usage examples

Single CLI binary `nba-sim` (defined in `src/nba_sim/cli.py` using Typer) with sub-commands:

```bash
nba-sim fetch             --start-season 2000 --end-season 2024 [--refresh]
nba-sim build-features    [--seasons 2022,2023,2024]
nba-sim train             --config configs/train.yaml --model {baseline|hierarchical} [--max-epochs N]
nba-sim evaluate          --split {val|test} --checkpoint models/best.pt
nba-sim simulate          --home BOS --away LAL --date 2025-02-14 [--n-samples 500] [--seed 42]
nba-sim cache-stats
nba-sim cache-clear       --endpoint boxscoretraditionalv3
```

Scripts in `scripts/` are one-liners that invoke the CLI with a preset config, meant for SLURM/cron:

- `scripts/fetch_all_seasons.py` → `nba-sim fetch --start-season 2000 --end-season 2024`
- `scripts/build_features.py` → `nba-sim build-features`
- `scripts/train.py` → `nba-sim train --config configs/train.yaml --model hierarchical`
- `scripts/simulate_game.py` → thin wrapper around `simulate_game(...)` with argparse for shell usage

### End-to-end from empty repo

```bash
pip install -e ".[dev]"
cp .env.example .env
nba-sim fetch --start-season 2000 --end-season 2024
nba-sim build-features
nba-sim train --config configs/train.yaml --model baseline
nba-sim train --config configs/train.yaml --model hierarchical
nba-sim evaluate --split test --checkpoint models/best.pt
nba-sim simulate --home BOS --away LAL --date 2025-02-14 --seed 42
```

---

## 9. Testing strategy

One test file per concern; markers for slow/network/GPU. Run `pytest -m "not slow and not network"` in CI.

| File | Covers |
| --- | --- |
| `tests/test_data_schema.py` | Pydantic schemas reject malformed rows; `MIN "mm:ss"` parser; FGM ≤ FGA validator; round-trip write/read of interim parquet preserves every field. |
| `tests/test_features.py` | No rolling feature uses future data (hypothesis property test); rolling-mean correctness on a hand-built fixture; NaN handling for cold-start players; leakage test across the train/val boundary. |
| `tests/test_model_shapes.py` | For a random-init model, forward-pass shapes match §4.2; gradient flows through every parameter (param.grad is not None after backward); embedding tables have the right size. |
| `tests/test_constraints.py` | Hypothesis-driven: sample 1,000 box scores from a random-init model; assert minutes sum to 240 (±0.1 after rounding), FGM ≤ FGA, 3PM ≤ 3PA, FTM ≤ FTA, 3PM ≤ FGM, all counts non-negative integers, team PTS equals sum of player PTS. |
| `tests/test_simulate_api.py` | `simulate_game` returns schema-valid `BoxScore`; determinism (same seed → same output); ensemble mode interval coverage sanity; unseen-player path doesn't crash and produces non-degenerate output. |

Property-based tests use `hypothesis` with shrinkers on roster size (1..15), player feature ranges, and sample counts.

---

## 10. Milestones

Each phase ends with specific passing tests + artifacts. No starting the next phase until the previous one is green.

### Phase 1 — Data pipeline + baseline (gate: beats "season-average" baseline)

- Implement `data/fetch.py`, `data/scrape_bref.py`, `data/schema.py`, `data/etl.py`, `data/splits.py`.
- Implement `features/rolling.py`, `features/context.py`, `features/matchup.py`.
- Implement `models/baseline_glm.py`.
- Implement enough of `cli.py` for `fetch`, `build-features`, `train --model baseline`, `evaluate`.
- **Ships when:** all `tests/test_data_schema.py` and `tests/test_features.py` pass; Poisson GLM beats "season-average" on every counting stat on validation season.

### Phase 2 — NN matches baseline (gate: NN within 5% of GLM per-stat MAE)

- Implement `models/encoders.py`, `models/heads.py`, `models/hierarchical.py`, `models/losses.py`.
- Implement `training/dataset.py`, `training/loop.py`, `training/metrics.py`.
- **Ships when:** `tests/test_model_shapes.py` and `tests/test_constraints.py` pass; hierarchical model trains to within 5% of GLM MAE on validation.

### Phase 3 — NN beats baseline (gate: meets §1 success criteria)

- Hyperparameter tuning on validation split. Adjust loss weights, regularization, curriculum.
- Ablations: no curriculum, no partial pooling, no attention pooling (replace with mean).
- **Ships when:** every §1 success criterion is met on validation. Test-set evaluation held back.

### Phase 4 — Simulation API + CLI

- Implement `simulate/sampler.py`, `simulate/api.py`.
- Wire `nba-sim simulate` in `cli.py`.
- Cold-start path; determinism harness.
- **Ships when:** `tests/test_simulate_api.py` passes; `nba-sim simulate --home BOS --away LAL --date 2025-02-14` prints a plausible box score in < 500 ms on CPU.

### Phase 5 — Polish, docs, packaging

- Run final test-set evaluation once. Commit the resulting `reports/` artifacts.
- Fill `README.md` gaps, check docstrings, type-check clean (`mypy` 0 errors), lint clean (`ruff check` 0 warnings).
- Tag `v0.1.0`.
- **Ships when:** `pytest && mypy && ruff check .` all clean; test-set metrics committed; release tagged.

---

## 11. Known risks and open questions

### Things v1 will handle

- DNPs and inactive players (zero-minute allocation).
- Traded players (per-game team membership).
- Rookies and new signings (cold-start embedding).
- Short-rotation games (the Dirichlet naturally concentrates mass).

### Things v1 will NOT handle (documented limitations)

- **In-game injuries.** If a star exits in Q1, v1 still assumes they play starter-level minutes. Could be added by a "minutes drawn from a truncated distribution conditional on a game-dependent gate" but that's v2.
- **Foul-outs.** PF count is modeled but doesn't feed back into a minutes cap. Expect occasional predictions with 8 PF — obviously wrong. Could be fixed with a joint constraint or a post-hoc projection for the PF-minutes pair.
- **Garbage-time / blowout rotations.** The model learns an average of both competitive and blowout distributions. Predictions for likely blowouts will be systematically wrong for end-of-bench players (under-predicted minutes). A "predicted competitiveness" feature could help; deferred to v2.
- **OT games.** MIN target is 240; OT games in training are flagged and either (a) excluded or (b) scaled to 240-equivalent. Decision: **exclude from training** in v1 (<5% of games), **predict as regulation** at inference time.
- **Playoff adjustments.** Playoff minutes distributions are tighter (starters play more, bench plays less). A `season_phase='playoffs'` feature is included but v1 won't explicitly model the regime shift.
- **Ejections.** Out of scope.
- **Starter vs. bench selection.** Model outputs minutes per rostered player; it does not distinguish "started" vs. "played off the bench." The output `PlayerBoxLine` has no `is_starter` field in v1.

### Open questions worth revisiting in v2

- Does modeling minutes as Dirichlet over the full 15-slot roster under-capacity bench players? Consider a zero-inflated Dirichlet or a mixture.
- Should pace and efficiency be modeled jointly (MVN) rather than independently? Correlation between pace and off_rtg is non-trivial.
- Is `nba_api` reliable enough for production inference (fetching current rosters)? If rate-limiting becomes painful, pin rosters to a nightly cached snapshot.
- Is Basketball Reference scraping worth maintaining once we're only modeling 2000+? Probably drop it after v1 unless a data gap appears.

---

## Appendix A — Stat list (modeled vs. derived)

**Modeled (15):** MIN, FGM, FGA, 3PM, 3PA, FTM, FTA, OREB, DREB, AST, STL, BLK, TOV, PF, plays-in-this-game gate.

**Derived at sampling time:**
- `PTS = 2*FGM + 3PM + FTM` — by definition, not sampled.
- `REB = OREB + DREB`.
- `+/-` from team margin × on-floor share.

**Team-level modeled (3):** pace, off_rtg (home), off_rtg (away). `def_rtg` of each team = opponent's `off_rtg` by definition.
