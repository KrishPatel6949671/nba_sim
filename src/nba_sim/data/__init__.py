"""Data acquisition, typing, and ETL.

Modules:
    fetch       : nba_api wrappers with caching + rate limiting.
    scrape_bref : Basketball Reference fallback for pre-nba_api coverage.
    schema      : Pydantic models for raw / interim / processed / box-score records.
    etl         : raw -> interim -> processed transforms.
    splits      : season-level train/val/test split construction.
"""
