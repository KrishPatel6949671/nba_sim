# NBA Box Score Simulator — Scaffolding & Implementation Plan

You are setting up a new Python project. **Do not implement any logic yet.** Your job in this pass is to:
1. Create the full directory structure with empty/stub files.
2. Write a comprehensive `PLAN.md` at the repo root describing the full implementation.
3. Write a `README.md` with project overview and setup instructions.
4. Create `pyproject.toml` (or `requirements.txt` — your call, justify it) with pinned dependencies.
5. Create a `.gitignore` appropriate for Python + ML projects (include data/, models/, .venv/, __pycache__, .ipynb_checkpoints, wandb/, etc.).

Stub files should contain a module docstring describing what will go in them and function/class signatures with `pass` or `raise NotImplementedError`. No real logic.

## Project context

- **Goal:** Production-usable NBA game simulator. Given two teams, rosters, and game context (home/away, rest, date), output a realistic full box score (~15 stats × ~10–13 players per team) with hard constraints respected: minutes sum to 240 per team, made ≤ attempted, counts are non-negative integers, team totals = sum of player contributions.
- **Stack:** Python 3.11+, PyTorch (CUDA), `nba_api` for data, Polars or Pandas for ETL (justify your choice in PLAN.md), DuckDB or Parquet for local storage.
- **Model:** Hierarchical neural network. Pace + offensive/defensive efficiency predicted at team level first, then allocated to players via attention-pooled roster encoder. Output heads parameterize distributions (Poisson/NegBinomial for counts, Beta for percentages, Dirichlet/softmax×240 for minutes allocation) so constraints are enforced by construction, not post-hoc projection. Use `torch.distributions` throughout for log-likelihood losses.
- **Training data:** ~25 seasons of box scores + rosters from `nba_api`, plus scraped supplemental data from Basketball Reference if needed for older seasons. Rolling N-game features (5/10/20), rest days, back-to-back flags, opponent defensive rating vs. position, usage trends. Learned player and team embeddings.
- **Validation:** Hold out full seasons, not random games, to avoid leakage through roster continuity and player form.
- **Baseline:** Poisson GLM / ridge regression baseline the NN must beat. Plan for this explicitly.
- **Serving:** The trained model should be usable from a simple Python API (`simulate_game(home_team, away_team, date, ...) -> BoxScore`) and a CLI. No web service required in v1 but structure the code so adding FastAPI later is trivial.

## Directory structure I want (adjust if you have a strong reason)

```
nba-sim/
├── README.md
├── PLAN.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── train.yaml
├── data/                      # gitignored, structure documented in README
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/                 # exploratory only, not part of pipeline
├── src/nba_sim/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetch.py           # nba_api wrappers with caching + rate limiting
│   │   ├── scrape_bref.py     # Basketball Reference fallback
│   │   ├── schema.py          # dataclasses / pydantic models for typed records
│   │   ├── etl.py             # raw -> interim -> processed
│   │   └── splits.py          # season-level train/val/test splits
│   ├── features/
│   │   ├── __init__.py
│   │   ├── rolling.py         # rolling N-game windows
│   │   ├── context.py         # rest, b2b, travel, season phase
│   │   └── matchup.py         # opponent def rating vs. position, etc.
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_glm.py    # Poisson GLM baseline
│   │   ├── encoders.py        # player encoder, roster set encoder w/ attention
│   │   ├── heads.py           # distribution-parameterizing output heads
│   │   ├── hierarchical.py    # full team->player hierarchical model
│   │   └── losses.py          # composite NLL loss across heads
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py         # torch Dataset / DataLoader
│   │   ├── loop.py            # training loop, eval, checkpointing
│   │   └── metrics.py         # per-stat calibration, box-score-level metrics
│   ├── simulate/
│   │   ├── __init__.py
│   │   ├── sampler.py         # sample a box score from the model
│   │   └── api.py             # simulate_game(...) public entry point
│   ├── cli.py                 # typer/click CLI: fetch, build-features, train, simulate
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── seed.py
│       └── io.py
├── scripts/
│   ├── fetch_all_seasons.py
│   ├── build_features.py
│   ├── train.py
│   └── simulate_game.py
└── tests/
    ├── test_data_schema.py
    ├── test_features.py
    ├── test_model_shapes.py
    ├── test_constraints.py    # minutes sum to 240, made<=attempted, etc.
    └── test_simulate_api.py
```

## What `PLAN.md` must cover

Structure it with clear top-level sections. Be specific — no hand-wavy "train the model" bullets. For each phase, list files touched, concrete deliverables, and how you'll know it works (what test passes, what metric moves, what artifact lands on disk).

Sections:

1. **Problem statement and success criteria.** What counts as "production-usable"? Define specific metrics: per-stat MAE vs. baseline, calibration (predicted vs. actual distributions), constraint violation rate (must be 0), inference latency target.

2. **Data pipeline.** Exact `nba_api` endpoints to hit (leaguegamefinder, boxscoretraditionalv2, boxscoreadvancedv2, commonteamroster, playergamelog, etc.). Rate limiting strategy. Caching layer (what's cached where, cache invalidation). Raw schema. Interim schema. Processed training schema. How Basketball Reference fills gaps. How to handle traded players, two-way contracts, DNPs, inactive lists.

3. **Feature engineering.** Complete feature list grouped by entity (player, team, matchup, context). For each, the exact computation. Leakage risks and how splits prevent them.

4. **Model architecture.** Tensor shapes at every stage. Player encoder dims, roster attention mechanism, team-level pace/efficiency head, player allocation head, per-stat distribution heads. Which `torch.distributions` class per stat and why. How constraints are enforced by construction. Parameter count target.

5. **Training.** Loss composition and weighting strategy. Optimizer, scheduler, batch size, epochs, early stopping. Curriculum if any (e.g., pretrain team-level head first). Regularization including partial pooling for low-sample players. Mixed precision, gradient clipping, checkpointing, experiment tracking (W&B vs. local TensorBoard — pick one).

6. **Evaluation.** Season-held-out validation protocol. Per-stat metrics. Calibration plots. Baseline comparison. Diagnostic: does it beat "predict the player's season average"? Does it beat the Poisson GLM?

7. **Simulation API.** Exact function signature for `simulate_game`. How to handle unseen players (rookies, signings). Sampling strategy (single sample vs. N samples with mean + interval). Determinism controls (seed).

8. **CLI and usage examples.** Concrete commands users will run, in order, from empty repo to simulated game.

9. **Testing strategy.** What each test file covers. Property-based tests for constraints (hypothesis).

10. **Milestones.** Ordered phases with clear gates. Phase 1: data pipeline + Poisson baseline. Phase 2: NN matches baseline. Phase 3: NN beats baseline. Phase 4: simulation API + CLI. Phase 5: polish, docs, packaging. Each phase ends with specific passing tests.

11. **Known risks and open questions.** Be honest about what's uncertain — minutes allocation realism, foul-outs, blowout garbage time, injuries mid-game. Flag what v1 will and won't handle.

## Rules

- No implementation. Stubs only. Every stub file should make it obvious what goes there.
- `PLAN.md` is the primary deliverable of this pass — treat it as the spec you'd hand to a contractor.
- Pin dependency versions. If you're unsure of the latest, note it as `# TODO: verify latest` rather than guessing.
- When you finish, print a summary of what you created and flag any decisions you made that I should review (Polars vs. Pandas, W&B vs. TensorBoard, pyproject vs. requirements, anything else).
- Ask me at most 3 clarifying questions *before* you start if anything material is unclear. Otherwise proceed.
