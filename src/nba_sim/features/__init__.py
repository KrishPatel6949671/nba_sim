"""Feature engineering — rolling windows, game context, matchup features.

Modules:
    rolling : N-game rolling aggregates (mean, rate-per-minute) for players and teams.
    context : game-context features (rest, travel, b2b, season phase, arena altitude).
    matchup : opponent-aware features (def rating vs position, head-to-head history).
"""
