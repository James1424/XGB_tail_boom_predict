"""Feature registry for ETF/index-holdings tail-event panel.

The first groups mirror the dense feature set from the previous XG boom project.
New groups add ETF source metadata, liquidity/size proxies, and volatility amplitude/frequency.
Future and label columns are intentionally excluded from FEATURE_GROUPS.
"""

FEATURE_GROUPS = {
    "core_momentum": [
        "mom_4m", "mom_5m", "mom_6m",
        "core_mom_456_avg", "core_mom_456_min", "core_mom_456_max", "core_mom_456_std",
        "mom_4m_vs_6m", "mom_5m_vs_6m",
        "mom_6m_first3m", "mom_6m_last3m", "mom_6m_acceleration",
    ],
    "sequence_path": [
        "ret_lag_1m", "ret_lag_2m", "ret_lag_3m", "ret_lag_4m", "ret_lag_5m", "ret_lag_6m",
        "positive_month_ratio_6m", "ret_lag_6m_mean", "ret_lag_6m_std", "ret_lag_6m_min", "ret_lag_6m_max",
    ],
    "relative_strength": [
        "rel_mom_1m_vs_qqq", "rel_mom_3m_vs_qqq", "rel_mom_6m_vs_qqq", "rel_mom_12m_vs_qqq",
    ],
    "other_momentum": [
        "mom_1m", "mom_2m", "mom_3m", "mom_7m", "mom_9m", "mom_12m", "mom_3m_vs_6m",
    ],
    "trend": [
        "price_ma5_ratio", "price_ma10_ratio", "price_ma20_ratio", "price_ma30_ratio", "price_ma50_ratio", "price_ma100_ratio",
        "ma5_slope_1m", "ma10_slope_1m", "ma20_slope_1m", "ma30_slope_1m", "ma50_slope_1m", "ma100_slope_1m",
    ],
    "risk_drawdown": [
        "drawdown_1m", "drawdown_3m", "drawdown_6m", "drawdown_12m",
        "drawdown_1m_abs", "drawdown_3m_abs", "drawdown_6m_abs", "drawdown_12m_abs",
        "volatility_1m", "volatility_3m", "volatility_6m",
        "return_vol_ratio_1m", "return_vol_ratio_3m", "return_vol_ratio_6m",
    ],
    "vol_structure": [
        "volatility_1m_to_3m", "volatility_1m_to_6m", "volatility_3m_to_6m",
        "atr_14_to_price", "atr_14_to_100d",
    ],
    "drawdown_recovery": [
        "recovery_from_1m_low", "recovery_from_3m_low", "recovery_from_6m_low",
        "drawdown_change_1m", "drawdown_change_3m", "days_since_3m_high_norm", "days_since_3m_low_norm",
    ],
    "volume_flow": [
        "volume_change_1m", "volume_change_3m", "volume_ratio_3m",
        "volume_ma1_to_6m", "volume_ma3_to_12m",
        "dollar_volume_change_1m", "dollar_volume_change_3m",
        "up_day_volume_ratio_3m", "up_day_dollar_volume_ratio_3m",
    ],
    "qqq_context": [
        "qqq_mom_1m", "qqq_mom_3m", "qqq_mom_6m", "qqq_mom_12m",
    ],
    "etf_source": [
        "source_count", "source_weight_sum", "theme_count",
        "in_core_growth", "in_large_cap_core", "in_large_cap_growth", "in_small_mid_growth",
        "in_semiconductor_ai", "in_software_cloud", "in_cybersecurity", "in_innovation_high_beta",
        "in_biotech", "in_clean_energy", "in_uranium_nuclear", "in_blockchain_fintech", "in_manual_core",
    ],
    "liquidity_size_proxy": [
        "avg_dollar_volume_1m", "avg_dollar_volume_3m", "avg_dollar_volume_6m",
        "log_avg_dollar_volume_3m", "dollar_volume_3m_to_6m",
        "trading_day_count_3m", "trading_day_count_6m",
    ],
    "volatility_amplitude_frequency": [
        "avg_abs_daily_return_1m", "avg_abs_daily_return_3m", "avg_abs_daily_return_6m",
        "large_move_freq_3m", "large_move_freq_6m",
        "up_big_move_freq_3m", "up_big_move_freq_6m",
        "down_big_move_freq_3m", "down_big_move_freq_6m",
        "intraday_range_mean_3m", "intraday_range_mean_6m",
        "liquid_vol_score",
    ],
}

LABEL_COLUMNS = [
    "future_return_1m", "future_return_2m", "future_return_3m", "future_max_return_1_3m",
    "future_max_return_1_3m_pct_rank", "monthly_top10_threshold_1_3m", "monthly_top5_threshold_1_3m",
    "label_top10_1_3m", "label_top5_1_3m", "label_boom30_top10_1_3m",
    "label_boom40_top10_1_3m", "label_boom50_top5_1_3m", "label_mega100_1_3m",
]

META_COLUMNS = ["month", "ticker", "adj_close"]


def all_declared_features() -> list[str]:
    out, seen = [], set()
    for features in FEATURE_GROUPS.values():
        for f in features:
            if f not in seen:
                out.append(f)
                seen.add(f)
    return out


def feature_group_name(feature: str) -> str:
    for group, features in FEATURE_GROUPS.items():
        if feature in features:
            return group
    return "unknown"
