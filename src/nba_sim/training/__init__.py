"""Training plumbing.

Modules:
    dataset : torch Dataset / DataLoader reading processed parquet.
    loop    : train() entry point — loss composition, curriculum, checkpointing.
    metrics : per-stat MAE, calibration, constraint-violation counters.
"""
