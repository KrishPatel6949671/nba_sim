"""Model components.

Modules:
    baseline_glm : Poisson GLM + Beta regression baseline that the NN must beat.
    encoders     : PlayerEncoder, RosterAttentionPool, GameContextEncoder.
    heads        : distribution-parameterizing output heads (Dirichlet/NB/Binomial/Normal).
    hierarchical : HierarchicalBoxScoreModel — team then player.
    losses       : composite NLL across heads with per-stat weighting.
"""
