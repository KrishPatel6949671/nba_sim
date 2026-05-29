# nba-sim v2 — Implementation Plan

This is the contractor-grade spec for the v2 work that builds on top of v1
(see `PLAN.md` for the v1 spec). v1 ships a simulator that requires the
target game to already exist in the processed parquets. **v2 lets users
simulate games that have not happened yet** — arbitrary future matchups,
no `--date` required, using rosters fetched live from the NBA API.

> **Conventions used in this document**
> - Same as `PLAN.md`: file paths are relative to the repo root; tensor
>   shapes are `[B, P, D]`; "Ships when" is the test(s) and artifact(s)
>   that gate completion.
> - Section numbering continues from `PLAN.md` (v1 ends at §11). The first
>   v2 phase is **Phase 6** (PLAN ended at Phase 5).

---

## 12. Problem statement and v2 success criteria

### Problem

In v1, the simulator looks up `(home, away, date)` in the processed
parquets and uses the actual roster + features the ETL already computed.
That gates everything on the game existing in the dataset. v2 removes
the gate:

- A user wants to simulate **next Tuesday's LAL @ PHX game** using each
  team's current roster and form, without that game being in the
  training data.
- The user runs one `nba-sim refresh` periodically to pull fresh rosters
  from `nba_api`, then `nba-sim simulate --home LAL --away PHX` with no
  date works against the snapshot.

### v2 confirmed scope (user-selected)

Per design discussion before this document was written, the v2 implementation
commits to:

- **(1a) Roster source: live `nba_api` fetch, cached.** The `refresh`
  command hits `CommonTeamRoster` for every NBA team and writes the
  result to `data/snapshot/rosters.parquet`. Cached on disk so repeated
  `simulate` invocations don't re-fetch.
- **(2b) Features as-of: last available game date by default.** The
  refresh's reference date defaults to *the day after the most recent
  game in `data/interim/`* — i.e., features are honest about what data
  we actually have, not what's true in calendar reality. An `--as-of
  YYYY-MM-DD` override is provided for explicit dating.
- **(3) Context features computed from as-of date + per-team last-game
  lookup.** `rest_days`, `b2b`, `travel_miles_prev`, `season_phase`,
  `day_of_week`, `month` are all derived mechanically from the as-of
  date and each team's most recent game in interim data.

### Success criteria

A run counts as v2-shippable when all of:

| Criterion | Target | Measurement |
| --- | --- | --- |
| `nba-sim simulate --home BOS --away LAL` (no `--date`) | returns a constraint-valid `BoxScore` | `tests/test_simulate_api.py` |
| Snapshot path produces same constraint guarantees as v1 | 0% violation rate on 100 ensemble samples | `tests/test_simulate_api.py::test_simulate_snapshot_constraints` |
| Snapshot freshness round-trip | `refresh` writes provenance JSON; `simulate` reads + validates | `tests/test_snapshot.py::test_snapshot_provenance_round_trip` |
| Refresh latency | ≤ 30 s for full 30-team refresh (rate-limited at 0.6 s/req) | wall-clock in `test_snapshot.py::test_refresh_under_30s[slow]` |
| Inference latency (snapshot path) | ≤ 750 ms on CPU including snapshot read | timed in `test_simulate_api.py` |
| Backward compat | v1 path (`--date` → matching parquet row) still works unchanged | existing tests stay green |
| Feature equivalence at known dates | When `as_of_date` matches a real game date in val, the snapshot-built feature row for that team is bit-equivalent (within float noise) to the processed parquet row | `tests/test_snapshot.py::test_snapshot_features_match_processed` |

Non-goals for v2:
- In-game injuries / DNP-because-injured (still v3).
- Custom roster overrides via `--home-roster "LeBron James,Anthony Davis,..."`
  (deferred to v2.1).
- Live game feeds / mid-game state updates.
- Foul-outs as a hard minutes cap.

---

## 13. Snapshot data layer

### 13.1 On-disk structure

A new directory mirrors `data/interim/` / `data/processed/`:

```
data/
└── snapshot/
    ├── rosters.parquet          # one row per (team_id, player_id) active in as_of season
    ├── player_features.parquet  # one row per (team_id, player_id) — feature vector for the synthetic future game
    ├── team_features.parquet    # one row per team_id — team rolling features as-of
    ├── team_lastgame.parquet    # per-team (last_game_date, last_arena_team_id) — used to derive rest_days / travel
    └── as_of.json               # provenance — date, refreshed_at, source, row counts
```

All four parquets share the convention "the as_of date is implicit in the
file; refresh atomically replaces the whole directory."

### 13.2 `as_of.json` schema

```json
{
  "as_of_date": "2024-03-15",
  "refreshed_at": "2024-03-16T08:42:11Z",
  "source": "nba_api+interim",
  "n_teams": 30,
  "n_players": 451,
  "interim_latest_game_date": "2024-03-14",
  "code_version": "v0.2.0-dev",
  "model_checkpoint_used_for_validation": null
}
```

`source` is `"nba_api+interim"` when both live and historical sources
contributed, `"interim-only"` if refresh was offline. The model
checkpoint field is set if `refresh` was followed by a sanity-check
forward pass.

### 13.3 `rosters.parquet` schema

| Column | Type | Notes |
| --- | --- | --- |
| `team_id` | i64 | NBA stats internal id |
| `team_abbr` | str | e.g. "BOS" |
| `player_id` | i64 | same id space as interim data |
| `player_name` | str | display name |
| `position` | str | from `CommonTeamRoster.POSITION` |
| `height_in` | f64 | parsed from `HEIGHT` ("6-9" → 81) |
| `weight_lbs` | f64 | from `WEIGHT` |
| `jersey` | str | nullable |
| `experience_years` | i64 | nullable |
| `birth_date` | date | nullable |
| `two_way` | bool | true for two-way contracts |

### 13.4 `player_features.parquet` schema

Mirrors the `_PLAYER_NUMERIC_COLS` / `_POSITION_TOKENS` / `_PLAYER_BOOL_COLS`
that `BoxScoreDataset` consumes, **per active rostered player**. The
critical wrinkle vs. processed parquet: every row's *target stats*
(`pts`, `fgm`, etc.) are absent — at inference these only feed teacher
forcing, which we skip. Schema is the strict subset that the model
*inputs* require.

### 13.5 `team_features.parquet` schema

One row per team containing the standardized team-rolling features
(`t_pace_5`, `t_pace_10`, `t_off_rtg_5`, `t_off_rtg_10`, `t_def_rtg_5`,
`t_def_rtg_10`, `t_win_pct_10`, `t_pts_avg_10`, `t_pts_allowed_10`).
Joined into the matchup vector at simulate time.

### 13.6 `team_lastgame.parquet` schema

| Column | Type | Notes |
| --- | --- | --- |
| `team_id` | i64 | |
| `last_game_date` | date | most recent game in interim with team_id participating |
| `last_arena_team_id` | i64 | the team whose arena hosted that game — used for travel miles |
| `last_was_home` | bool | for downstream sanity (rest days don't care; travel does) |

### 13.7 Ships when

- `pytest tests/test_snapshot.py::test_schema_validity -q` passes.
- A round-trip refresh → read produces parquets that match the documented
  schemas exactly (column names, dtypes, non-null constraints).

---

## 14. Refresh pipeline

`src/nba_sim/snapshot/refresh.py` orchestrates the build. Three stages:

### 14.1 Stage A — Resolve `as_of_date`

Pseudo:
```
if --as-of YYYY-MM-DD is given:
    as_of_date = YYYY-MM-DD
else:
    as_of_date = latest_interim_game_date + 1 day  # "the day after the most recent data we have"
```

Read latest game date from `data/interim/<latest_season>/games.parquet`.
Surface the resolved date to the user before any nba_api call so they
can abort if it looks wrong.

### 14.2 Stage B — Fetch rosters from `nba_api`

For each of the 30 NBA teams (use the canonical team_id list from
`nba_api.stats.static.teams.get_teams()`):

```
roster = CommonTeamRoster(team_id=t, season=current_season).get_data_frames()[0]
```

Honor the existing `fetch.cached_call` rate limiter (0.6 s/req
default — see `data/fetch.py`). Total ~20 s for cold-start, < 1 s when
fully cached.

Augment with `CommonPlayerInfo(player_id=...)` for each *new* player not
yet seen by `data/raw/.cache/` — used for height / weight / experience.

Validate via a new Pydantic model `RawRosterEntry` mirroring the
`CommonTeamRoster` columns. Errors → `qa_report.json` so a malformed
roster doesn't poison the rest.

### 14.3 Stage C — Build features as-of

This is the heart of v2. The existing `rolling.py` / `matchup.py` /
`context.py` modules already handle "compute features for one row given
all prior rows." We exploit that by **constructing a synthetic target row**
per (team, player) with `date = as_of_date`, then running the existing
feature functions over `interim + synthetic`. The synthetic-row outputs
are exactly the features the model would have seen if the game had
been played on `as_of_date`.

#### Player feature assembly

```
# Pseudo-Polars
all_player_box = pl.concat([
    pl.read_parquet(f"data/interim/{s}/player_box.parquet")
    for s in seasons_up_to(as_of_date)
])
all_games = pl.concat([
    pl.read_parquet(f"data/interim/{s}/games.parquet")
    for s in seasons_up_to(as_of_date)
])

synthetic_target_box, synthetic_target_games = _build_synthetic_target(
    rosters, as_of_date,
)

combined_box = pl.concat([all_player_box, synthetic_target_box])
combined_games = pl.concat([all_games, synthetic_target_games])

p_rolling = player_rolling(combined_box, combined_games, team_box=all_team_box)
p_std = season_to_date(combined_box, combined_games)

# Extract just the synthetic-target rows
player_features = (
    p_rolling.join(p_std, on=["game_id", "player_id"], how="left")
             .filter(pl.col("game_id").str.starts_with("SNAPSHOT_"))
)
```

The synthetic target rows carry sentinel `game_id`s like
`SNAPSHOT_<team_abbr>_<as_of>` and `date == as_of_date` so the
filter-back step is unambiguous.

#### Team feature assembly

Same approach: append a synthetic team-box row per team with `game_id =
SNAPSHOT_TEAM_<team_abbr>_<as_of>`, run `team_rolling`, filter to the
synthetic rows.

#### `team_lastgame.parquet`

Computed inline from `all_games`: per team, find the row with `date <
as_of_date` and max date.

### 14.4 Stage D — Atomic write

Write each parquet to `data/snapshot/.tmp/<name>.parquet`, then rename
the whole `.tmp/` directory to `data/snapshot/`. Same idempotency
discipline as the v1 ETL.

`as_of.json` is written last so partial refreshes never advertise as
complete.

### 14.5 CLI surface

```bash
nba-sim refresh                              # as-of = latest_interim + 1 day
nba-sim refresh --as-of 2024-03-15           # explicit
nba-sim refresh --force                      # bypass "snapshot newer than interim" skip
nba-sim refresh --offline                    # skip nba_api fetch; reuse last rosters.parquet
```

`--offline` exists for development without network (CI; flaky travel
days) — it reads the previous `rosters.parquet` and only recomputes
features.

### 14.6 Ships when

- `nba-sim refresh --as-of 2023-04-01` (a known-good date in held-out
  interim data) writes the four parquets and `as_of.json` with row
  counts matching the documented expectations.
- For 5 random `(team, player)` pairs whose first appearance is on or
  after `as_of_date`, the synthetic feature row's `p_min_avg_10`
  matches a hand-computed value from the interim parquet within 1e-6.
- `pytest tests/test_snapshot.py -q` passes.

---

## 15. Snapshot-to-game-row assembly

When `simulate_game(home, away, date=None)` is called, the dataset
layer needs a single row per active player on each side, matching the
schema `BoxScoreDataset` expects (everything in `_PLAYER_NUMERIC_COLS`
+ position + bool flags + matchup + context).

### 15.1 New module: `src/nba_sim/snapshot/build.py`

Public function:

```python
def build_synthetic_game_batch(
    *,
    home_team: str,
    away_team: str,
    snapshot_dir: Path = Path("data/snapshot"),
    train_parquet: Path = Path("data/processed/train.parquet"),
    max_players: int = 15,
) -> tuple[dict[str, torch.Tensor], list[tuple[int, str]], list[tuple[int, str]], str]:
    """Build a (model_batch, home_players, away_players, as_of_iso) tuple.

    1. Read snapshot/rosters.parquet + player_features.parquet
       + team_features.parquet + team_lastgame.parquet + as_of.json.
    2. Filter rosters to (home, away).
    3. Sort each side by p_min_avg_10 desc to mirror the training-time
       roster ordering.
    4. Assemble per-side feature blocks identical to BoxScoreDataset._build_side.
    5. Assemble context vector from as_of_date + team_lastgame lookups.
    6. Assemble matchup vector from team_features.
    7. Adopt train_parquet's player_id_map + feature_stats for
       standardization — same train-vs-eval discipline as in v1.
    """
```

The output `model_batch` is a `dict[str, torch.Tensor]` with `B == 1`
that goes straight into `model.forward(batch)`. No teacher-forcing keys.

### 15.2 Context vector specifics

Compose from snapshot/team_lastgame + as_of_date:

| Feature | Source |
| --- | --- |
| `is_home` | True for home roster rows, False for away |
| `rest_days` | `(as_of_date - team_lastgame.last_game_date).clip(0, 5)` |
| `b2b` | `rest_days == 1` |
| `is_3in4`, `is_4in6` | look back into interim games for that team |
| `season_phase` | from `as_of_date`: `"early"` (Oct–Nov), `"mid"` (Dec–Jan), `"late"` (Feb–Apr), `"playoffs"` (May+) |
| `day_of_week` | `as_of_date.weekday()` |
| `month` | `as_of_date.month` |
| `altitude_ft` | lookup home team's arena |
| `travel_miles_prev` | great-circle between `team_lastgame.last_arena` arena and home team's arena, per team |

The lookups for `altitude_ft` / arena coordinates are already in
`features/context.py::arena_altitude` / `travel_distance_miles`.

### 15.3 Matchup vector specifics

Identical to `BoxScoreDataset._build_matchup`, but the team-rolling
stats come from `snapshot/team_features.parquet` instead of the
processed parquet's per-row team columns. The 16-dim layout is
unchanged.

### 15.4 Cold-start fallback (tier 1 + tier 3 only for v2)

Unseen `player_id` → maps to embedding id 0 (pad/unknown) — same as v1.
The three-tier projection (PLAN §7.3 tier 2) requires training an
auxiliary attribute-projection head and is deferred to v2.1.

This means a brand-new signing on refresh day will get the generic
"unknown player" embedding. Acceptable for v2; the model's positional
features still inform the prediction non-trivially.

### 15.5 Ships when

- `tests/test_snapshot.py::test_build_synthetic_game_batch_shapes` —
  output batch has the right tensor shapes and dtypes.
- `tests/test_snapshot.py::test_build_synthetic_game_batch_no_nans` —
  no NaNs propagate past standardization; cold-start players get
  finite numeric features (means or zeros via `FeatureStats`).
- `tests/test_snapshot.py::test_context_vector_matches_v1_at_known_date` —
  for a date that exists in val.parquet, the snapshot-built context
  vector matches the v1 context vector to within float noise.

---

## 16. Simulation API v2 changes

### 16.1 Signature

```python
def simulate_game(
    home_team: str,
    away_team: str,
    date: str | _dt.date | None = None,    # CHANGED: now optional
    *,
    home_roster: list[str] | None = None,
    away_roster: list[str] | None = None,
    n_samples: int = 1,
    seed: int | None = None,
    device: str = "auto",
    checkpoint: str | Path = "models/best.pt",
    train_parquet: str | Path = "data/processed/train.parquet",
    val_parquet: str | Path = "data/processed/val.parquet",
    test_parquet: str | Path | None = "data/processed/test.parquet",
    snapshot_dir: str | Path = "data/snapshot",   # NEW
    return_distributions: bool = False,
) -> BoxScore | BoxScoreEnsemble:
    ...
```

`date=None` is the new default. `home_roster` / `away_roster` /
`return_distributions=True` remain v2.1+ deferrals.

### 16.2 Branching logic

```
if date is None:
    use snapshot path (build_synthetic_game_batch + sample)
elif date in val_parquet or test_parquet:
    use v1 path (existing)
else:
    raise LookupError(...)
```

A `--strict-snapshot` flag (or `force_snapshot=True` kwarg) bypasses
the v1 lookup entirely — useful for verifying snapshot quality on
known dates.

### 16.3 Date object on the output

`BoxScore.date` becomes the `as_of_date` from the snapshot when the
snapshot path is used. The `BoxScore.seed` field is unchanged.

### 16.4 Errors

| Error | When | Message |
| --- | --- | --- |
| `FileNotFoundError` | snapshot_dir doesn't exist or is missing files | `"run 'nba-sim refresh' first"` |
| `ValueError` | snapshot's `as_of.json` is older than 7 days | warning, not fail; user override via `--stale-ok` |
| `LookupError` | requested `home_team` or `away_team` not in roster snapshot | `"team XXX not in snapshot — try nba-sim refresh"` |

### 16.5 Ships when

- `simulate_game(home="LAL", away="PHX", date=None)` returns a valid
  `BoxScore` when snapshot exists.
- All existing v1 simulate_api tests still pass (backwards compat).
- New tests:
  - `test_simulate_snapshot_path_returns_valid_boxscore`
  - `test_simulate_snapshot_deterministic_with_seed`
  - `test_simulate_no_snapshot_raises_clear_error`
  - `test_simulate_v1_path_still_works_with_date`
  - `test_simulate_force_snapshot_overrides_v1_lookup`

---

## 17. CLI v2 changes

### 17.1 New / changed subcommands

```bash
nba-sim refresh [--as-of YYYY-MM-DD] [--force] [--offline]
nba-sim simulate --home BOS --away LAL [--date YYYY-MM-DD] [--force-snapshot] [--stale-ok]
nba-sim snapshot-status         # NEW: print as_of.json contents in human form
```

### 17.2 `nba-sim snapshot-status`

```
$ nba-sim snapshot-status
as_of_date:                    2024-03-15
refreshed_at:                  2024-03-16T08:42:11Z (yesterday)
source:                        nba_api+interim
teams:                         30
players:                       451
interim_latest_game_date:      2024-03-14
freshness:                     fresh (1 day old)
```

Prints a clean human-readable status block so users know when to
refresh.

### 17.3 Ships when

- `nba-sim refresh --help` renders cleanly.
- `nba-sim simulate --home LAL --away PHX` (no date) works against a
  refreshed snapshot, printing the same box-score format as v1.
- `nba-sim snapshot-status` prints a parseable status block.

---

## 18. Testing strategy

### 18.1 New file: `tests/test_snapshot.py`

| Test | Covers |
| --- | --- |
| `test_resolve_as_of_date_default` | uses latest_interim + 1 day when --as-of absent |
| `test_resolve_as_of_date_explicit` | --as-of override is respected |
| `test_refresh_writes_all_four_parquets` | offline mode + tiny fixture interim data |
| `test_refresh_provenance_json_complete` | as_of.json has every documented key |
| `test_refresh_atomic_partial_write_safety` | crash mid-write doesn't leave bad files visible |
| `test_synthetic_target_rows_use_sentinel_ids` | game_id starts with SNAPSHOT_ |
| `test_player_feature_matches_processed_at_known_date` | bit-equivalent within float noise |
| `test_context_vector_rest_days_correct` | rest_days = as_of - last_game_date |
| `test_build_synthetic_game_batch_shapes` | model batch shapes match v1 |
| `test_build_synthetic_game_batch_no_nans` | unseen player → finite features |
| `test_refresh_under_30s` (slow, network) | full 30-team refresh ≤ 30 s |

### 18.2 Extensions to `tests/test_simulate_api.py`

| New test | Purpose |
| --- | --- |
| `test_simulate_snapshot_path_returns_valid_boxscore` | end-to-end no-date sim |
| `test_simulate_snapshot_deterministic_with_seed` | same seed → identical |
| `test_simulate_no_snapshot_raises_clear_error` | helpful error msg |
| `test_simulate_force_snapshot_overrides_v1_lookup` | flag works |
| `test_simulate_snapshot_minutes_sum_to_240` | hard constraint preserved |
| `test_simulate_v1_path_still_works_with_date` | backwards-compat |

### 18.3 Network / artifact guards

All snapshot tests use `_require_artifacts` (snapshot_dir exists) +
`_require_network` (when hitting nba_api) skip helpers, same pattern as
v1's `tests/test_simulate_api.py`. CI without artifacts still passes.

### 18.4 Ships when

- All v1 tests still pass.
- All new tests pass on a local environment with a refreshed snapshot.
- Fast suite (no `slow` or `network` marker) runs in CI without artifacts.

---

## 19. Milestones (Phase 6 onward)

Each phase ends with specific passing tests + artifacts. v1's Phase 5
was the polish/tag of v0.1.0; v2 starts at Phase 6.

### Phase 6 — Snapshot data layer + refresh pipeline

- Implement `src/nba_sim/snapshot/__init__.py`, `rosters.py`, `refresh.py`.
- Implement `nba-sim refresh` CLI subcommand.
- Implement `nba-sim snapshot-status` CLI subcommand.
- **Ships when:** `tests/test_snapshot.py` (non-network subset) passes;
  `nba-sim refresh --offline --as-of <date>` writes all four parquets +
  `as_of.json` against a fixture interim directory; `nba-sim
  snapshot-status` prints the documented block format.

### Phase 7 — Snapshot-to-game-row + simulate v2

- Implement `src/nba_sim/snapshot/build.py::build_synthetic_game_batch`.
- Extend `simulate_game` signature with optional `date` and `snapshot_dir`.
- Implement branching logic (snapshot path vs v1 path).
- Wire `nba-sim simulate` to drop the `--date` requirement.
- **Ships when:** new tests in §18.2 pass; running `nba-sim simulate
  --home LAL --away PHX` (no date) against a refreshed snapshot prints a
  box score in < 750 ms on CPU.

### Phase 8 — Network integration + cache discipline

- Wire real `nba_api.CommonTeamRoster` calls behind
  `fetch.cached_call`.
- Add `--refresh` flag that bypasses cache for current-season endpoints
  (consistent with v1 `nba-sim fetch --refresh`).
- Property test: refresh twice in a row with cached responses takes
  < 1 s (cache hit) and produces identical output.
- **Ships when:** `nba-sim refresh` (no `--offline`) successfully
  completes against the live API on a clean cache; output matches the
  offline-mode equivalent for the same as-of date.

### Phase 9 — Cold-start tier 2 (optional within v2 — could slip to v2.1)

- Add `models/projection.py`: a small MLP that maps
  `(height, weight, position, experience, age)` → `player_embedding`.
- Add an auxiliary reconstruction loss to the training pipeline:
  `L_proj = ||proj(attrs_i) - embedding_i||²` averaged over seen
  players.
- At simulate time, unseen players whose attributes are in
  `commonplayerinfo` get the projected embedding instead of id 0.
- **Ships when:** rookie-cold-start MAE on a synthetic injected-rookie
  benchmark is ≥ 5% better than the v1 "id=0" fallback; existing
  trained checkpoints remain loadable (the new head is a strict
  addition with default-bypass).

### Phase 10 — Polish + docs + tag

- Update `README.md` with the snapshot tutorial.
- Update `PLAN.md` cross-references (mark sections "v1, see v2PLAN.md
  §X for evolution").
- Tag `v0.2.0`.
- **Ships when:** `pytest && mypy && ruff check .` all clean; README's
  new tutorial is runnable end-to-end; tag is on a commit that passes
  CI.

---

## 20. Known risks and v3 open questions

### Things v2 will handle

- Future games (any date past `interim_latest_game_date`).
- Trade-deadline roster moves (refresh picks up live).
- Two-way contracts (flagged in roster, treated as regular players).
- Pre-game rest day / b2b context derived from each team's actual
  recent schedule.

### Things v2 will NOT handle (documented limitations)

- **Injury reports / probable-out lists.** Refresh fetches the full
  signed roster from `CommonTeamRoster`. If LeBron is listed
  questionable for tonight, he's still in the roster and the
  simulator gives him starter minutes. v3 candidate: pull from
  `playerinfo` or a dedicated injury endpoint and pass an `out_for`
  flag.
- **Live game state.** Simulator runs once at the as-of date; it
  doesn't update mid-game. Streaming support is v3+.
- **Custom roster strings.** `home_roster=["LeBron James", ...]`
  remains v2.1 — needs name → player_id resolution and feature
  composition for arbitrary roster subsets.
- **Mean-mode return.** `simulate_game(..., mode="mean")` remains
  deferred; the current ensemble path provides functionally
  equivalent info.

### Open questions worth revisiting in v3

- **Is the as-of-date offset of +1 day correct?** Sometimes a team's
  next game is several days out — using `as_of_date = latest_interim_date
  + 1` may overstate rest days. Consider: `as_of_date = next_scheduled_game_for(home_team)`,
  using `ScoreboardV2` for the schedule. Trade-off: more accurate but
  introduces a schedule-aware dependency.
- **Should snapshot include per-player injury / out-for status?** If we
  add an `is_active` bool to `rosters.parquet`, the simulator can
  trivially mask injured players. Source TBD — the nba_api injury
  endpoints are flaky.
- **Lineup-aware features.** The model currently sees a flat roster;
  on-court / off-court interactions are averaged away. A v3
  enhancement: condition the player allocation on lineup probabilities
  from the team's recent rotation patterns.
- **Tier-2 cold start without retraining.** The projection head needs
  a training pass to be useful. If we want it without retraining,
  consider a simpler **positional centroid** fallback: project
  unseen players onto the closest training-time embedding by
  Euclidean distance in `(height, weight, position)` space.

---

## Appendix B — Synthetic target row construction (v2 internal)

A worked example: simulating LAL @ PHX as-of 2024-03-15, with two
players on LAL (LeBron #2544, AD #203076) and one on PHX (Durant
#201142):

```python
synthetic_target_box = pl.DataFrame([
    # LAL synthetic targets
    {"game_id": "SNAPSHOT_LAL_2024-03-15", "player_id": 2544, "team_id": 1610612747,
     "minutes": 0, "pts": 0, ...all stat cols 0..., "position": "F"},
    {"game_id": "SNAPSHOT_LAL_2024-03-15", "player_id": 203076, "team_id": 1610612747,
     "minutes": 0, "pts": 0, ..., "position": "F"},
    # PHX synthetic targets
    {"game_id": "SNAPSHOT_PHX_2024-03-15", "player_id": 201142, "team_id": 1610612756,
     "minutes": 0, "pts": 0, ..., "position": "F"},
])

synthetic_target_games = pl.DataFrame([
    {"game_id": "SNAPSHOT_LAL_2024-03-15", "date": date(2024, 3, 15),
     "season": 2023, "home_team_id": 1610612747, "away_team_id": 1610612756,
     "home_pts": 0, "away_pts": 0, "dropped": False},
    {"game_id": "SNAPSHOT_PHX_2024-03-15", "date": date(2024, 3, 15),
     "season": 2023, "home_team_id": 1610612756, "away_team_id": 1610612747,
     "home_pts": 0, "away_pts": 0, "dropped": False},
])
```

Two `SNAPSHOT_` games are needed (not one) because `team_rolling` and
`player_rolling` both consume one row per team — having both teams in
the same game would either double-count or require special handling.
The duplication is paid once per refresh and only affects the synthetic
rows; the rolling computations don't see the duplicates because each
team's rolling features look at their own history independently.

The zero-filled targets (`pts=0`, etc.) are harmless — they're never
fed to the model at inference (teacher-forcing keys are stripped) and
not used in any rolling window because rolling windows look strictly
*before* the target row.

---

## Appendix C — Code organization

```
src/nba_sim/snapshot/
├── __init__.py        # public exports: refresh, build_synthetic_game_batch
├── rosters.py         # nba_api fetch + caching
├── refresh.py         # orchestrator: as_of_date → rosters → features → write
├── build.py           # build_synthetic_game_batch
└── status.py          # snapshot-status CLI helper
```

CLI additions in `src/nba_sim/cli.py`:
- `refresh` subcommand
- `snapshot-status` subcommand
- `simulate` flags: `--date` now optional, `--force-snapshot`, `--stale-ok`

No new dependencies — everything reuses `nba_api`, `polars`, `pydantic`,
`typer`, and `torch` from v1's pinned set.
