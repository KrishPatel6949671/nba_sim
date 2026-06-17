# nba-sim

An NBA box score simulator built on a hierarchical neural network. Given two teams — with an explicit date, or **no date at all to simulate a game that hasn't been played yet** — it produces a realistic full box score: per-player minutes, points, shooting splits, rebounds, assists, steals, blocks, turnovers, fouls, and plus-minus, plus per-team pace and offensive / defensive ratings.

Every sampled box score satisfies the hard constraints of basketball **by construction** — not by post-hoc projection:

- Team minutes sum to exactly 240
- `FGM ≤ FGA`, `3PM ≤ 3PA`, `FTM ≤ FTA`, `3PM ≤ FGM`
- All counts are non-negative integers
- `team_pts = Σ player_pts`, `REB = OREB + DREB`

This is achieved by choosing output distributions that are correct by their math: `Dirichlet(α) × 240` for minutes allocation, `Binomial(FGA, p_make)` for makes-given-attempts, `NegativeBinomial` for counting stats. The model never has to learn the constraints — they're carried by the parameterization.

See [PLAN.md](PLAN.md) for the full design and build plan.

## Status

**v0.2 (dev) — v2 Phases 6–8 complete.** The full pipeline (fetch → features → train → evaluate → simulate) works end-to-end, and the v2 **snapshot layer** now lets you simulate games that haven't been played yet — no `--date` required — using rosters pulled live from the NBA API and cached on disk. The remaining v2 work (cold-start attribute projection, the `v0.2.0` tag) is tracked in [v2PLAN.md](v2PLAN.md) §19.

| Component | Status |
| --- | --- |
| Data pipeline + Pydantic schemas | ✅ |
| Feature engineering (rolling, matchup, context) | ✅ |
| Poisson GLM baseline + Season-Average baseline | ✅ |
| Hierarchical NN (team head + player allocation head) | ✅ trained |
| Composite NLL + coupling loss + partial-pooling regularization | ✅ |
| Evaluation (per-stat MAE, calibration, constraint checks) | ✅ |
| Constraint-safe sampler (`sample_raw_box_score`, `sample_box_score`, `sample_ensemble`) | ✅ |
| `simulate_game(...)` + `nba-sim simulate` with `--date` (v1 path) | ✅ |
| **Live-roster snapshot layer (`nba-sim refresh`) + dateless simulation** | ✅ v2 (Phase 6–7) |
| **Live `nba_api` roster fetch + cache discipline (`--refresh`, `cache-stats`/`cache-clear`)** | ✅ v2 (Phase 8) |
| Active-rotation modeling (caps each side at ~10 active players) | ✅ v2 |
| Cold-start attribute projection (rookies → projected embedding) | ⚠ deferred (Phase 9 / v2.1) |
| Custom roster overrides (`--home-roster "..."`) | ⚠ deferred (v2.1) |
| Injury / out-for status (questionable players still get starter minutes) | ⚠ deferred (v3) |

## Results

Evaluated on the held-out **2024-25 season** (`reports/metrics.json`):

| Metric | Value |
| --- | --- |
| Points MAE (per player-game) | **3.71** |
| Minutes MAE | 3.74 |
| Rebounds MAE | 1.54 |
| Assists MAE | 1.08 |
| Team points MAE — team head | **9.28** (player-sum path: 12.88) |
| Pace MAE | 3.72 |
| Offensive-rating MAE | 8.84 |
| Constraint-violation rate (all 7 hard checks) | **0.000** |

Every one of the 7 hard constraints — minutes sum to 240, `FGM ≤ FGA`, `3PM ≤ 3PA`, `FTM ≤ FTA`, `3PM ≤ FGM`, non-negativity, and integer counts — holds on **100%** of sampled box scores, because they're guaranteed by construction rather than enforced after the fact.

## Example output

Real CLI output for a single sampled game:

```
$ nba-sim simulate --home LAL --away PHX --date 2023-10-26 --seed 42 --device cpu

[HOME] LAL: PTS=135  pace=102.5  off_rtg=107.4  def_rtg=111.8
  Player                   MIN  PTS      FG      3P      FT  REB  AST  STL  BLK  TOV
  LeBron James            43.0   34 14/28    2/2     4/4       4    3    1    0    1
  Austin Reaves           40.0   13  3/18    1/3     6/8       8    9    1    0    3
  D'Angelo Russell        39.4   23  8/19    2/6     5/7       5    5    1    0    2
  Anthony Davis           38.5   39 16/29    0/0     7/8      14    4    1    2    3
  Taurean Prince          26.2   14  5/11    2/6     2/2       2    1    2    0    0
  Christian Wood          14.9    1  0/0     0/2     1/2       5    1    0    0    1
  Rui Hachimura           14.1    0  0/4     0/2     0/0       1    0    1    0    1
  ...

[AWAY] PHX: PTS=85  pace=102.5  off_rtg=111.8  def_rtg=107.4
  Player                   MIN  PTS      FG      3P      FT  REB  AST  STL  BLK  TOV
  Kevin Durant            32.8   20  7/13    0/3     6/7       6    6    1    1    4
  Eric Gordon             37.2   18  9/18    0/3     0/0       7    2    0    2    0
  ...
```

Use `--n-samples 100` to get the ensemble mean + 10th/90th-percentile team totals instead of one draw.

## Requirements

- Python 3.11 or 3.12
- A CUDA-capable GPU is recommended for training (~13 minutes per epoch on RTX 3050; the code runs on CPU but ~50× slower)
- ~20 GB of disk for the full 25-season dataset + features + checkpoints

## Setup

This project is managed with [`uv`](https://github.com/astral-sh/uv). The commands below use `uv pip`; the `pip` equivalents (inside an activated venv) also work.

```bash
# 1. Clone
git clone <repo-url> nba-sim && cd nba-sim

# 2. Virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# 3. Install PyTorch with the right CUDA build FIRST
#    (pyproject.toml pins torch==2.5.1 but the installer needs the CUDA index to find the CUDA wheel)
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
#    CPU-only build:
#    uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# 4. Install the project + dev extras
uv pip install -e ".[dev]"

# 5. Copy the example env file
cp .env.example .env

# 6. Sanity check
pytest -q
nba-sim --help
```

> **Note:** `pyproject.toml` pins `click==8.1.8` on purpose — the pinned `typer==0.15.1` is incompatible with click ≥ 8.2 (it crashes `--help` rendering). Don't bump click without also bumping typer to ≥ 0.16.

## Tutorial — going from a fresh clone to a simulated game

The CLI commands chain together. You can stop after any step and pick up later — every step is idempotent and writes its output to disk.

### 1. Fetch raw box scores from nba_api

```bash
nba-sim fetch --start-season 2014 --end-season 2025
```

This downloads per-game traditional + advanced box scores, team rosters, and player metadata for every regular-season game in the range (`2025` is the 2025-26 season). Output lands in `data/interim/<season>/`.

**Expect this to take hours, not minutes.** The `nba_api` endpoints are rate-limited to ~one request every 0.6 seconds. The fetch is fully resumable — every API call is cached to `data/raw/.cache/`, so re-running will skip work already done. If `nba_api` returns empty payloads for older seasons, fall back to the Basketball Reference scraper by setting `scrape_bref.enabled: true` in `configs/data.yaml`.

### 2. Build features and processed splits

```bash
nba-sim build-features
```

This reads the per-season interim parquets, computes rolling and career features per player, attaches opponent/matchup/context features, and writes `data/processed/{train,val,test}.parquet`. The splits are configured in `configs/data.yaml` (default: train on the 2014-15 → 2023-24 seasons **plus 2025-26**, with the **2024-25 season held out** for validation/test).

All features use a strict "prior-only" leakage discipline: for any game on date `d`, every feature value is computed from games with `date < d`. The first game a player or team plays in any season produces null for every rolling feature; the model handles that with masking + imputation downstream.

### 3. Train the baselines (optional but recommended)

```bash
nba-sim train --config configs/train.yaml --model baseline
```

Trains a `PoissonGLMBaseline` (per-stat ridge-link Poisson regression for counts, linear regression for minutes). This is the bar the neural network must clear by ≥5–15% per-stat MAE on validation (PLAN §1). Runs in minutes on CPU.

### 4. Train the hierarchical neural network

```bash
nba-sim train --config configs/train.yaml --model hierarchical
```

Trains the main model end-to-end. On an RTX 3050: roughly 12–15 minutes per epoch, typically converging in 5–15 epochs (early-stopping with patience=10 on validation composite NLL). Writes:
- `models/best.pt` — best-epoch checkpoint by val NLL
- `models/ckpt_epoch_N.pt` — per-epoch checkpoints
- Training summary printed to stdout + saved alongside

Key features active by default:
- d_team=192, player MLP [192, 192] (tuned via `scripts/tune5.py`)
- Smooth-L1 coupling loss at λ=0.01 pulling `Σ E[player_pts]` toward the team head's `pace × off_rtg / 100` estimate
- Career-pooled shooting priors (`p_fg_pct_career`, `p_ft_pct_career`, etc.) and FT-rate features
- Opponent defensive matchup features by position (`opp_def_rtg_vs_pos`, `opp_blk_allowed_vs_pos`)

### 5. Evaluate on validation (or test)

```bash
# Validation (use during iteration)
nba-sim evaluate --split val --checkpoint models/best.pt --report-dir reports/val

# Test (run ONCE at the end — held out per PLAN §6.1)
nba-sim evaluate --split test --checkpoint models/best.pt --report-dir reports/test
```

Produces `metrics.json` with per-stat MAE / RMSE, 80% predictive interval coverage, team_pts MAE (both player-sum and team-head paths), pace and off_rtg MAE, and constraint-violation rate. Also writes reliability plots and a team-PTS scatter PNG.

Drop `--no-plots` for faster metrics-only passes.

### 6. Simulate a game

```bash
# Single sample, deterministic
nba-sim simulate --home LAL --away PHX --date 2023-10-26 --seed 42 --device cpu

# Ensemble of 100 samples — prints mean + 10th/90th percentile team totals
nba-sim simulate --home LAL --away PHX --date 2023-10-26 --n-samples 100 --seed 42
```

When you pass `--date`, the `(home, away, date)` tuple must correspond to a game present in your processed val or test parquet (the v1 path). To simulate a game that *hasn't* been played, drop `--date` and use the snapshot — see the next step.

### 7. Simulate a game that hasn't happened yet (snapshot path)

First build a local snapshot. `refresh` pulls every team's current roster live from `nba_api`, then derives each player's as-of features from the interim data:

```bash
nba-sim refresh                 # rosters live from nba_api; as-of = latest interim game + 1 day
nba-sim refresh --offline       # reuse the last rosters.parquet, only recompute features (no network)
nba-sim refresh --as-of 2026-04-14   # pin an explicit as-of date

nba-sim snapshot-status         # show the snapshot's as-of date, team/player counts, freshness
```

Then simulate with **no date** — the matchup is built from the snapshot, so it can be any future game:

```bash
nba-sim simulate --home BOS --away LAL                 # uses data/snapshot/
nba-sim simulate --home BOS --away LAL --n-samples 100 --seed 42
nba-sim simulate --home BOS --away LAL --date 2024-01-15 --force-snapshot   # snapshot path even with a date
```

The snapshot path caps each side at roughly its top-10 active players (an active-rotation model), so team totals stay realistic instead of summing all 15 rostered players. Rosters reflect whatever `nba_api` returns at refresh time, so trade-deadline moves are picked up automatically — but injury/out-for status is **not** modeled, so a questionable star still draws starter minutes (see [Known limitations](#known-limitations)).

The nba_api response cache is shared with `fetch`; inspect or prune it with `nba-sim cache-stats` and `nba-sim cache-clear`.

## Python API

```python
from nba_sim.simulate.api import simulate_game

# Dateless / snapshot path — simulate a game that hasn't been played.
# Requires a snapshot on disk (`nba-sim refresh` first).
bs = simulate_game(home_team="BOS", away_team="LAL", seed=42, device="cpu")
print(bs.date)                                # the snapshot's as_of_date

# Single-sample BoxScore against a known historical game (v1 path)
bs = simulate_game(
    home_team="LAL",
    away_team="PHX",
    date="2023-10-26",            # optional; omit it for the snapshot path
    n_samples=1,
    seed=42,
    device="cpu",
)
print(bs.home.team, bs.home.pts)              # "LAL" 135
print(bs.home.players[0].player_name)         # "LeBron James"
print(bs.home.players[0].pts, bs.home.players[0].minutes)  # 34, 43.0

# Ensemble — mean + 10th/90th percentile summaries per cell
ens = simulate_game(
    home_team="LAL", away_team="PHX", date="2023-10-26",
    n_samples=100, seed=42, device="cpu",
)
print(ens.mean.home.pts, ens.interval_low.home.pts, ens.interval_high.home.pts)
# e.g. 118, 102, 134

# Lower-level API: sample from an existing BoxScoreDistribution
from nba_sim.simulate.sampler import sample_box_score
# (assumes you already have a `dist` from model.forward(batch))
bs = sample_box_score(
    dist,
    home_players=[(2544, "LeBron James"), ...],
    away_players=[(201142, "Kevin Durant"), ...],
    home_team="LAL", away_team="PHX",
    date_iso="2023-10-26", seed=42,
)
```

All returned objects are Pydantic models with full type information — see `src/nba_sim/data/schema.py` (`BoxScore`, `BoxScoreEnsemble`, `SimTeamBoxLine`, `SimPlayerBoxLine`).

## CLI reference

| Command | Purpose |
| --- | --- |
| `nba-sim fetch --start-season N --end-season M` | Download box scores + rosters from nba_api into `data/interim/` |
| `nba-sim build-features` | Compute features and write `data/processed/{train,val,test}.parquet` |
| `nba-sim train --config configs/train.yaml --model {baseline,hierarchical}` | Train and checkpoint to `models/` |
| `nba-sim evaluate --split {val,test} --checkpoint MODELS --report-dir DIR` | Score a checkpoint, write `metrics.json` + plots |
| `nba-sim refresh [--as-of YYYY-MM-DD] [--offline] [--refresh] [--force]` | Build/refresh `data/snapshot/` from live nba_api rosters + as-of features |
| `nba-sim snapshot-status` | Print the snapshot's provenance (`as_of.json`) + freshness verdict |
| `nba-sim simulate --home ABC --away XYZ [--date YYYY-MM-DD] [--force-snapshot] [--n-samples N] [--seed K]` | Sample a box score (or ensemble); no `--date` → snapshot path |
| `nba-sim cache-stats` / `nba-sim cache-clear` | Inspect / prune the nba_api response cache |

Run any command with `--help` for full option list. `--verbose` / `-v` is available on most commands.

## Data layout

Everything under `data/` is gitignored. The pipeline creates:

```
data/
├── raw/         # unmodified API payloads (cached per-resource)
│   └── .cache/  # nba_api response cache (parquet)
├── interim/     # per-season typed parquet: games, player_box, team_box, rosters
├── processed/   # modeling-ready parquet: train/val/test splits with all features joined
└── snapshot/    # live-roster snapshot for dateless simulation (rosters, player/team
                 # features, team_lastgame, as_of.json) — written by `nba-sim refresh`
```

Checkpoints land under `models/`, evaluation artifacts under `reports/`.

## Known limitations

These are deliberate scope cuts, not bugs. PLAN §11 and v2PLAN §20 enumerate them in detail.

- **Single-sample minutes can exceed 48 per player.** The Dirichlet × 240 sampler produces team minutes summing to exactly 240, but individual players occasionally draw > 48 minutes when the Dirichlet mass is concentrated on them. Use ensembles + the mean if this matters; the issue averages out across draws.
- **No injury / out-for status on the snapshot path.** `refresh` pulls the full signed roster from `nba_api`. A player listed questionable for tonight is still in the snapshot and the simulator gives him starter minutes. Injury-aware masking is a v3 candidate.
- **Cold-start path is partial.** Unseen `player_id`s (e.g. brand-new signings) map to the embedding's pad/unknown bucket, so simulations don't crash, but the attribute-projection head from PLAN §7.3 / v2PLAN Phase 9 (rookie attributes → projected embedding) is deferred to v2.1.
- **Custom roster overrides** (`--home-roster "LeBron James,..."` / the `home_roster=` kwarg) are a v2.1 deferral — the snapshot uses each team's full live roster.
- **No in-game injuries, foul-outs, or OT dynamics.** The model learns averages over all game states. Predictions for likely blowouts will under-predict bench minutes; predictions for stars who actually foul out will over-predict their minutes.
- **`return_distributions=True`** kwarg on the Python API is a v2.1 placeholder.

## Development

```bash
# Linting + formatting
ruff check .
ruff format .

# Type checking
mypy

# Full test suite
pytest

# Tests are marked: slow, network, gpu. CI runs:
pytest -m "not slow and not network"
```

Tests live under `tests/` and follow the convention `test_<module>.py`. The simulate-API integration tests under `tests/test_simulate_api.py` conditionally skip when `models/best.pt` or the processed parquets aren't on disk, so a fresh clone won't fail the suite.

## Project layout

```
src/nba_sim/
├── data/         # nba_api fetchers, typed Pydantic schemas, ETL (raw → interim → processed)
├── features/     # rolling windows, matchup, context (rest, travel, altitude, day-of-week)
├── models/       # baseline GLM, encoders, distribution heads, hierarchical model, losses
├── training/     # Dataset/DataLoader, training loop, metrics, evaluation
├── simulate/     # constraint-safe sampler, public simulate_game(...) API
├── snapshot/     # live-roster refresh + as-of feature build for dateless simulation
├── utils/        # logging, seeding, IO helpers
└── cli.py        # typer CLI: fetch | build-features | train | evaluate | simulate | refresh | snapshot-status | cache-stats | cache-clear
```

Each module's docstring explains the contract it satisfies. See `PLAN.md` for the full architectural rationale.

## License

MIT

