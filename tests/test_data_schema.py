"""Schema-layer tests.

Covers:
    - Pydantic validators on :class:`PlayerBoxLine` (fgm <= fga, tpm <= fga, etc.).
    - ``"mm:ss"`` minutes parser.
    - Round-trip write/read of interim parquet preserves every field.
    - Raw -> interim coercion on synthetic malformed rows.
"""

from __future__ import annotations

import pytest


def test_player_box_rejects_fgm_exceeding_fga() -> None:
    pytest.skip("not implemented")


def test_player_box_rejects_tpm_exceeding_fgm() -> None:
    pytest.skip("not implemented")


def test_minutes_mm_ss_parser() -> None:
    pytest.skip("not implemented")


def test_interim_parquet_round_trip() -> None:
    pytest.skip("not implemented")


def test_raw_tolerates_missing_fields() -> None:
    pytest.skip("not implemented")


def test_dnp_implies_zero_minutes() -> None:
    pytest.skip("not implemented")
