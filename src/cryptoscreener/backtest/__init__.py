"""Offline backtest and evaluation module.

Implements evaluation metrics per PRD §10 and EVALUATION_METRICS.md:
- AUC, PR-AUC per horizon
- Brier score, ECE (Expected Calibration Error)
- Top-K capture rate
- Mean net_edge_bps in top-K
- Churn metrics (rank stability)
"""

from cryptoscreener.backtest.metrics import (
    BacktestMetrics,
    CalibrationMetrics,
    ChurnMetrics,
    TopKMetrics,
    compute_auc,
    compute_brier_score,
    compute_ece,
    compute_pr_auc,
    compute_topk_capture,
    compute_topk_mean_edge,
)

__all__ = [
    "BacktestMetrics",
    "CalibrationMetrics",
    "ChurnMetrics",
    "TopKMetrics",
    "compute_auc",
    "compute_brier_score",
    "compute_ece",
    "compute_pr_auc",
    "compute_topk_capture",
    "compute_topk_mean_edge",
]
