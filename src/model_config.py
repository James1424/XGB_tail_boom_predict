from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

# Preferred full panel path. In the integrated project, build_panel writes this file to outputs/.
PANEL_FILE = OUTPUT_DIR / "clean_monthly_panel.csv"
# Fallbacks keep local/manual usage possible.
PANEL_FALLBACKS = [
    DATA_DIR / "clean_monthly_panel.csv",
    PROJECT_ROOT / "outputs" / "panel_head_20000.csv",
    DATA_DIR / "panel_head_20000.csv",
]

TARGET_LABEL = "label_boom30_top10_1_3m"
AUX_LABELS = [
    "label_top10_1_3m",
    "label_top5_1_3m",
    "label_boom40_top10_1_3m",
    "label_boom50_top5_1_3m",
    "label_mega100_1_3m",
]
FUTURE_RETURN_COLS = ["future_return_1m", "future_return_2m", "future_return_3m", "future_max_return_1_3m"]
LEAKAGE_COLUMNS = FUTURE_RETURN_COLS + [
    "future_max_return_1_3m_pct_rank",
    "monthly_top10_threshold_1_3m",
    "monthly_top5_threshold_1_3m",
    TARGET_LABEL,
] + AUX_LABELS
ID_COLUMNS = ["month", "ticker", "adj_close", "sources", "categories"]

TRAIN_END = "2021-12-31"
VALID_START = "2022-01-01"
VALID_END = "2023-12-31"
TEST_START = "2024-01-01"

TOP_KS = [3, 5, 10]
MAIN_SEEDS = [7, 42, 202, 777, 2026]
ROUND_SEEDS = list(range(100))

BASE_MODEL_PARAMS = {
    "n_estimators": 650,
    "max_depth": 3,
    "learning_rate": 0.025,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "colsample_bylevel": 0.85,
    "min_child_weight": 8,
    "reg_alpha": 0.05,
    "reg_lambda": 3.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": -1,
}

ROUND_MODEL_PARAMS = dict(BASE_MODEL_PARAMS)
ROUND_MODEL_PARAMS.update({"n_estimators": 220, "learning_rate": 0.035})

ABLATION_GROUPS = {
    "core_momentum": ["mom_4m", "mom_5m", "mom_6m", "core_mom_456", "mom_6m_"],
    "other_momentum": ["mom_1m", "mom_2m", "mom_3m", "mom_7m", "mom_9m", "mom_12m"],
    "relative_strength": ["rel_mom_"],
    "trend": ["price_ma", "ma5_slope", "ma10_slope", "ma20_slope", "ma30_slope", "ma50_slope", "ma100_slope"],
    "risk_drawdown": ["drawdown", "volatility_", "return_vol_ratio"],
    "volatility_frequency": ["large_move_freq", "up_big_move_freq", "down_big_move_freq", "avg_abs_daily_return", "intraday_range"],
    "liquidity_size": ["avg_dollar_volume", "dollar_volume", "trading_day_count", "liquid_vol_score"],
    "volume_flow": ["volume_change", "volume_ratio", "volume_ma", "up_day_volume", "up_day_dollar"],
    "qqq_context": ["qqq_mom_"],
    "etf_source": ["source_count", "source_weight_sum", "theme_count", "in_"],
}

OUTPUT_FILES = {
    "main_result": OUTPUT_DIR / "reference_downweighted_main_model_result.csv",
    "strategy_baseline": OUTPUT_DIR / "strategy_baseline_comparison.csv",
    "latest_live": OUTPUT_DIR / "latest_live_boom_candidates.csv",
    "ablation": OUTPUT_DIR / "ablation_ranked_summary.csv",
    "five_seed": OUTPUT_DIR / "five_seed_training_stability.csv",
    "rounds_100": OUTPUT_DIR / "rounds_100_report.csv",
    "rounds_frequency": OUTPUT_DIR / "rounds_100_latest_candidate_frequency.csv",
    "feature_importance": OUTPUT_DIR / "feature_importance.csv",
    "monthly_top": OUTPUT_DIR / "monthly_top_predictions.csv",
    "full_predictions": OUTPUT_DIR / "full_predictions.csv",
    "metrics_json": OUTPUT_DIR / "model_metrics.json",
}
MAIN_MODEL_FILE = MODEL_DIR / "xgb_tail_event_classifier.json"
FEATURE_LIST_FILE = MODEL_DIR / "selected_features.txt"
README_FILE = PROJECT_ROOT / "README.md"
