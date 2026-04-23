# nba-sim

A production-usable NBA box score simulator. Given two teams, rosters, and game context (home/away, rest days, date), it outputs a realistic full box score — ~15 stats per player for ~10–13 players per team — with hard constraints respected:

- minutes sum to 240 per team,
- made ≤ attempted for every shot / FT stat,
- counts are non-negative integers,
- team totals equal the sum of player contributions.

Constraints are enforced **by construction** via the choice of output distributions (Dirichlet × 240 for minutes allocation, Poisson/NegBinomial for counts, Beta for percentages), not by post-hoc projection.

See [PLAN.md](PLAN.md) for the full design and build plan. This README covers setup and usage only.

## Status

Scaffold. No implementation yet — every module is a typed stub. Follow the milestones in `PLAN.md`.

## Requirements

- Python 3.11 or 3.12
- A CUDA-capable GPU is recommended for training (the code runs on CPU but ~100× slower)
- ~20 GB of disk for the full 25-season dataset + features + checkpoints

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url> nba-sim && cd nba-sim

# 2. Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch with the right CUDA build FIRST
#    (pyproject.toml pins torch==2.5.1 but pip needs the CUDA index to find the CUDA wheel)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
#    For CPU-only:
#    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# 4. Install the project in editable mode with dev extras
pip install -e ".[dev]"

# 5. Copy the example env file and fill in values
cp .env.example .env

# 6. Sanity-check
pytest -q
nba-sim --help
```

## Data layout

Everything under `data/` is gitignored. The pipeline creates this structure:

```
data/
├── raw/         # unmodified API payloads and Basketball Reference HTML, cached per-resource
│   └── .cache/  # nba_api response cache (parquet)
├── interim/     # per-season typed parquet: games.parquet, player_box.parquet, team_box.parquet, rosters.parquet
└── processed/   # modeling-ready parquet: train/val/test splits with rolling features joined in
```

## Usage

The CLI is the primary entry point. Typical flow:

```bash
# Fetch ~25 seasons of box scores + rosters (slow, rate-limited — hours, not minutes)
nba-sim fetch --start-season 2000 --end-season 2024

# Build features and write processed splits
nba-sim build-features

# Train the baseline Poisson GLM (fast, minutes)
nba-sim train --config configs/train.yaml --model baseline

# Train the hierarchical NN
nba-sim train --config configs/train.yaml --model hierarchical

# Simulate a single game
nba-sim simulate --home BOS --away LAL --date 2025-02-14 --seed 42

# Simulate N samples and report mean + 80% interval
nba-sim simulate --home BOS --away LAL --date 2025-02-14 --n-samples 500
```

Scripts in `scripts/` are thin wrappers around the CLI for batch / cluster runs.

## Python API

```python
from nba_sim.simulate.api import simulate_game

box = simulate_game(
    home_team="BOS",
    away_team="LAL",
    date="2025-02-14",
    n_samples=1,
    seed=42,
)
print(box.home.players)  # list[PlayerBoxLine]
```

See `PLAN.md` §7 for the full signature and unseen-player handling.

## Development

```bash
ruff check .
ruff format .
mypy
pytest
pytest -m "not slow and not network"   # fast suite
```

Tests are split by marker (`slow`, `network`, `gpu`). CI runs the fast suite on every push.

## Project layout

```
src/nba_sim/
├── data/         # nba_api + Basketball Reference fetchers, typed schemas, ETL
├── features/     # rolling windows, rest/travel context, matchup features
├── models/       # baseline GLM, encoders, distribution heads, hierarchical model, losses
├── training/     # Dataset/DataLoader, training loop, metrics
├── simulate/     # sampling, public simulate_game(...) API
├── utils/        # logging, seeding, IO helpers
└── cli.py        # typer CLI: fetch | build-features | train | simulate
```

See `PLAN.md` for the full design, and each stub file's module docstring for what goes where.

## License

MIT
