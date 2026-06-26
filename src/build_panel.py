import numpy as np
import pandas as pd

from .config import (
    DAILY_PRICES_FILE, SEED_UNIVERSE_FILE, RAW_PANEL_FILE, CLEAN_PANEL_FILE,
    FEATURE_MISSING_REPORT_FILE, DROPPED_FEATURES_FILE, FEATURE_MANIFEST_FILE,
    PANEL_SUMMARY_FILE, LABEL_SUMMARY_FILE, LABEL_BY_MONTH_FILE, LATEST_PANEL_SAMPLE_FILE,
    MODEL_DESIGN_SAMPLE_FILE, PANEL_HEAD_20000_FILE, OUTPUT_DIR, FIRST_SAMPLE_MONTH, BENCHMARK,
    MAX_FEATURE_MISSING_RATE, MIN_FEATURE_NON_NULL_ROWS,
    TAIL_TOP10_Q, TAIL_TOP5_Q, BOOM30, BOOM40, BOOM50, MEGA100,
    LARGE_MOVE_ABS_THRESHOLD, UP_BIG_MOVE_THRESHOLD, DOWN_BIG_MOVE_THRESHOLD,
)
from .feature_groups import all_declared_features, feature_group_name, LABEL_COLUMNS, META_COLUMNS


def _month_end_idx(df: pd.DataFrame) -> pd.Index:
    return df.groupby(["ticker", "month"])["date"].idxmax()


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def _prepare_daily(prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        if c in px.columns:
            px[c] = pd.to_numeric(px[c], errors="coerce")
    if "adj_close" not in px.columns and "close" in px.columns:
        px["adj_close"] = px["close"]
    for c in ["open", "high", "low", "close"]:
        if c not in px.columns:
            px[c] = px["adj_close"]
    px = px.dropna(subset=["date", "ticker", "adj_close"])
    px["ticker"] = px["ticker"].astype(str).str.upper()
    px["daily_ret"] = px.groupby("ticker")["adj_close"].pct_change()
    px["abs_daily_ret"] = px["daily_ret"].abs()
    px["large_move_day"] = (px["abs_daily_ret"] >= LARGE_MOVE_ABS_THRESHOLD).astype(float)
    px["up_big_move_day"] = (px["daily_ret"] >= UP_BIG_MOVE_THRESHOLD).astype(float)
    px["down_big_move_day"] = (px["daily_ret"] <= DOWN_BIG_MOVE_THRESHOLD).astype(float)
    px["dollar_volume"] = px["adj_close"] * px["volume"]
    px["intraday_range"] = (px["high"] - px["low"]).abs() / px["adj_close"].replace(0, np.nan)
    px["month"] = px["date"].dt.to_period("M").dt.to_timestamp("M")
    return px


def _days_since_extreme(s: pd.Series, window: int, is_high: bool) -> pd.Series:
    def f(x):
        arr = np.asarray(x, dtype=float)
        if np.all(np.isnan(arr)):
            return np.nan
        idx = np.nanargmax(arr) if is_high else np.nanargmin(arr)
        return len(arr) - 1 - idx
    return s.rolling(window, min_periods=max(30, window // 2)).apply(f, raw=False)


def _daily_month_end_features(prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.copy().sort_values(["ticker", "date"])
    g = px.groupby("ticker", group_keys=False)

    # Moving averages and slopes.
    for w in [5, 10, 20, 30, 50, 100]:
        px[f"ma{w}"] = g["adj_close"].transform(lambda s: s.rolling(w, min_periods=w).mean())
        px[f"ma{w}_prev21"] = g[f"ma{w}"].shift(21)
        px[f"ma{w}_slope_1m"] = px[f"ma{w}"] / px[f"ma{w}_prev21"] - 1

    # Rolling highs/lows/drawdowns/recoveries/volatility.
    windows = [(21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")]
    for days, name in windows:
        minp = max(15, days // 2)
        px[f"high_{name}"] = g["adj_close"].transform(lambda s: s.rolling(days, min_periods=minp).max())
        px[f"low_{name}"] = g["adj_close"].transform(lambda s: s.rolling(days, min_periods=minp).min())
        px[f"drawdown_{name}"] = px["adj_close"] / px[f"high_{name}"] - 1
        px[f"recovery_from_{name}_low"] = px["adj_close"] / px[f"low_{name}"] - 1
        if name in ["1m", "3m", "6m"]:
            px[f"volatility_{name}"] = g["daily_ret"].transform(lambda s: s.rolling(days, min_periods=minp).std() * np.sqrt(252))
            px[f"avg_abs_daily_return_{name}"] = g["abs_daily_ret"].transform(lambda s: s.rolling(days, min_periods=minp).mean())
        if name in ["3m", "6m"]:
            px[f"large_move_freq_{name}"] = g["large_move_day"].transform(lambda s: s.rolling(days, min_periods=minp).mean())
            px[f"up_big_move_freq_{name}"] = g["up_big_move_day"].transform(lambda s: s.rolling(days, min_periods=minp).mean())
            px[f"down_big_move_freq_{name}"] = g["down_big_move_day"].transform(lambda s: s.rolling(days, min_periods=minp).mean())
            px[f"intraday_range_mean_{name}"] = g["intraday_range"].transform(lambda s: s.rolling(days, min_periods=minp).mean())
            px[f"avg_dollar_volume_{name}"] = g["dollar_volume"].transform(lambda s: s.rolling(days, min_periods=minp).mean())
            px[f"trading_day_count_{name}"] = g["adj_close"].transform(lambda s: s.rolling(days, min_periods=1).count())
    px["avg_dollar_volume_1m"] = g["dollar_volume"].transform(lambda s: s.rolling(21, min_periods=15).mean())
    px["log_avg_dollar_volume_3m"] = np.log1p(px["avg_dollar_volume_3m"])
    px["dollar_volume_3m_to_6m"] = px["avg_dollar_volume_3m"] / px["avg_dollar_volume_6m"].replace(0, np.nan)

    # ATR-like range features.
    prev_close = g["adj_close"].shift(1)
    true_range = pd.concat([
        (px["high"] - px["low"]).abs(),
        (px["high"] - prev_close).abs(),
        (px["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    px["true_range"] = true_range
    px["atr_14"] = g["true_range"].transform(lambda s: s.rolling(14, min_periods=10).mean())
    px["atr_100"] = g["true_range"].transform(lambda s: s.rolling(100, min_periods=50).mean())
    px["atr_14_to_price"] = px["atr_14"] / px["adj_close"]
    px["atr_14_to_100d"] = px["atr_14"] / px["atr_100"].replace(0, np.nan)

    px["days_since_3m_high_norm"] = g["adj_close"].transform(lambda s: _days_since_extreme(s, 63, True)) / 63.0
    px["days_since_3m_low_norm"] = g["adj_close"].transform(lambda s: _days_since_extreme(s, 63, False)) / 63.0

    # Volume flow features.
    px["up_volume"] = np.where(px["daily_ret"] > 0, px["volume"], np.nan)
    px["down_volume"] = np.where(px["daily_ret"] < 0, px["volume"], np.nan)
    px["up_dollar_volume"] = np.where(px["daily_ret"] > 0, px["dollar_volume"], np.nan)
    px["down_dollar_volume"] = np.where(px["daily_ret"] < 0, px["dollar_volume"], np.nan)
    px["avg_up_volume_3m"] = g["up_volume"].transform(lambda s: s.rolling(63, min_periods=20).mean())
    px["avg_down_volume_3m"] = g["down_volume"].transform(lambda s: s.rolling(63, min_periods=20).mean())
    px["avg_up_dollar_volume_3m"] = g["up_dollar_volume"].transform(lambda s: s.rolling(63, min_periods=20).mean())
    px["avg_down_dollar_volume_3m"] = g["down_dollar_volume"].transform(lambda s: s.rolling(63, min_periods=20).mean())
    px["up_day_volume_ratio_3m"] = px["avg_up_volume_3m"] / px["avg_down_volume_3m"].replace(0, np.nan)
    px["up_day_dollar_volume_ratio_3m"] = px["avg_up_dollar_volume_3m"] / px["avg_down_dollar_volume_3m"].replace(0, np.nan)

    idx = _month_end_idx(px)
    cols = [
        "ticker", "month", "adj_close", "volume", "dollar_volume",
        "avg_dollar_volume_1m", "avg_dollar_volume_3m", "avg_dollar_volume_6m", "log_avg_dollar_volume_3m", "dollar_volume_3m_to_6m",
        "trading_day_count_3m", "trading_day_count_6m",
        "drawdown_1m", "drawdown_3m", "drawdown_6m", "drawdown_12m",
        "recovery_from_1m_low", "recovery_from_3m_low", "recovery_from_6m_low",
        "volatility_1m", "volatility_3m", "volatility_6m",
        "avg_abs_daily_return_1m", "avg_abs_daily_return_3m", "avg_abs_daily_return_6m",
        "large_move_freq_3m", "large_move_freq_6m", "up_big_move_freq_3m", "up_big_move_freq_6m",
        "down_big_move_freq_3m", "down_big_move_freq_6m", "intraday_range_mean_3m", "intraday_range_mean_6m",
        "atr_14_to_price", "atr_14_to_100d",
        "days_since_3m_high_norm", "days_since_3m_low_norm",
        "up_day_volume_ratio_3m", "up_day_dollar_volume_ratio_3m",
    ]
    for w in [5, 10, 20, 30, 50, 100]:
        cols += [f"ma{w}", f"ma{w}_slope_1m"]
    return px.loc[idx, [c for c in cols if c in px.columns]].copy()


def _attach_source_features(panel: pd.DataFrame) -> pd.DataFrame:
    if not SEED_UNIVERSE_FILE.exists():
        return panel
    seed = pd.read_csv(SEED_UNIVERSE_FILE)
    seed["ticker"] = seed["ticker"].astype(str).str.upper()
    source_cols = [c for c in seed.columns if c.startswith("in_")]
    keep = ["ticker", "source_count", "source_weight_sum", "theme_count"] + source_cols
    keep = [c for c in keep if c in seed.columns]
    out = panel.merge(seed[keep], on="ticker", how="left")
    for c in keep:
        if c != "ticker":
            out[c] = out[c].fillna(0)
    return out


def _add_tail_labels(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    valid = out["future_max_return_1_3m"].notna()
    g = out[valid].groupby("month")["future_max_return_1_3m"]
    out["future_max_return_1_3m_pct_rank"] = out.groupby("month")["future_max_return_1_3m"].rank(pct=True, ascending=True)
    out["monthly_top10_threshold_1_3m"] = np.nan
    out["monthly_top5_threshold_1_3m"] = np.nan
    thresholds = g.quantile([TAIL_TOP10_Q, TAIL_TOP5_Q]).unstack()
    thresholds.columns = ["top10", "top5"]
    out = out.merge(thresholds, left_on="month", right_index=True, how="left")
    out["monthly_top10_threshold_1_3m"] = out["top10"]
    out["monthly_top5_threshold_1_3m"] = out["top5"]
    out = out.drop(columns=["top10", "top5"], errors="ignore")
    out["label_top10_1_3m"] = ((out["future_max_return_1_3m"] >= out["monthly_top10_threshold_1_3m"]) & valid).astype(int)
    out["label_top5_1_3m"] = ((out["future_max_return_1_3m"] >= out["monthly_top5_threshold_1_3m"]) & valid).astype(int)
    out["label_boom30_top10_1_3m"] = ((out["label_top10_1_3m"] == 1) & (out["future_max_return_1_3m"] >= BOOM30)).astype(int)
    out["label_boom40_top10_1_3m"] = ((out["label_top10_1_3m"] == 1) & (out["future_max_return_1_3m"] >= BOOM40)).astype(int)
    out["label_boom50_top5_1_3m"] = ((out["label_top5_1_3m"] == 1) & (out["future_max_return_1_3m"] >= BOOM50)).astype(int)
    out["label_mega100_1_3m"] = ((out["future_max_return_1_3m"] >= MEGA100) & valid).astype(int)
    return out


def _compute_liquid_vol_score(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    # Cross-sectional ranks by month. This is not a label and uses only current/past features.
    rank_specs = {
        "avg_dollar_volume_3m": 0.35,
        "volatility_6m": 0.25,
        "large_move_freq_6m": 0.20,
        "up_big_move_freq_6m": 0.20,
    }
    score = pd.Series(0.0, index=out.index)
    for c, w in rank_specs.items():
        if c in out.columns:
            r = out.groupby("month")[c].rank(pct=True, ascending=True)
            score = score + w * r.fillna(0)
    out["liquid_vol_score"] = score
    return out


def _filter_missing(panel: pd.DataFrame, candidate_features: list[str]) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    available = [c for c in candidate_features if c in panel.columns]
    stats = []
    for c in available:
        non_null = int(panel[c].notna().sum())
        missing_count = int(panel[c].isna().sum())
        missing_rate = float(panel[c].isna().mean())
        first_valid = panel.loc[panel[c].notna(), "month"].min() if non_null else pd.NaT
        last_valid = panel.loc[panel[c].notna(), "month"].max() if non_null else pd.NaT
        dropped = missing_rate > MAX_FEATURE_MISSING_RATE or non_null < MIN_FEATURE_NON_NULL_ROWS
        reason = []
        if missing_rate > MAX_FEATURE_MISSING_RATE:
            reason.append(f"missing_rate>{MAX_FEATURE_MISSING_RATE:.0%}")
        if non_null < MIN_FEATURE_NON_NULL_ROWS:
            reason.append(f"non_null_rows<{MIN_FEATURE_NON_NULL_ROWS}")
        stats.append({
            "feature": c,
            "feature_group": feature_group_name(c),
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "non_null_rows": non_null,
            "non_null_rate": 1 - missing_rate,
            "first_valid_month": first_valid,
            "last_valid_month": last_valid,
            "dropped": dropped,
            "drop_reason": ";".join(reason),
        })
    report = pd.DataFrame(stats).sort_values(["dropped", "missing_rate"], ascending=[False, False])
    dropped = report[report["dropped"]].copy()
    kept_features = report.loc[~report["dropped"], "feature"].tolist()
    keep_cols = META_COLUMNS + kept_features + [c for c in LABEL_COLUMNS if c in panel.columns]
    keep_cols = [c for c in keep_cols if c in panel.columns]
    return panel[keep_cols].copy(), kept_features, report, dropped


def _write_manifest(panel: pd.DataFrame, kept_features: list[str], dropped: pd.DataFrame) -> None:
    lines = [
        "# ETF Tail Panel Feature Manifest",
        "",
        "This file lists model input features kept after missingness filtering.",
        "Future-return and tail-label columns are targets/evaluation fields and must not be used as model input features.",
        "",
        f"Rows: {len(panel):,}",
        f"Columns: {len(panel.columns):,}",
        f"Tickers: {panel['ticker'].nunique():,}",
        f"Months: {panel['month'].nunique():,}",
        f"Kept features: {len(kept_features):,}",
        f"Dropped features: {len(dropped):,}",
        "",
        "## Kept model features",
    ]
    for c in kept_features:
        lines.append(f"- {c} [{feature_group_name(c)}]")
    lines.append("\n## Target / label columns")
    for c in LABEL_COLUMNS:
        if c in panel.columns:
            lines.append(f"- {c}")
    lines.append("\n## Dropped high-missing features")
    if dropped.empty:
        lines.append("_No features dropped by missingness filter._")
    else:
        for _, r in dropped.iterrows():
            lines.append(f"- {r['feature']}: missing_rate={r['missing_rate']:.2%}, non_null_rows={int(r['non_null_rows'])}, reason={r['drop_reason']}")
    FEATURE_MANIFEST_FILE.write_text("\n".join(lines), encoding="utf-8")


def _sample_with_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "sample_source", source)
    return out


def _write_model_design_sample(clean: pd.DataFrame) -> None:
    parts = []
    if clean.empty:
        pd.DataFrame().to_csv(MODEL_DESIGN_SAMPLE_FILE, index=False)
        return

    panel = clean.copy()
    panel["month"] = pd.to_datetime(panel["month"])

    # 1) Latest rows: helpful for checking what a live prediction month looks like.
    latest_month = panel["month"].max()
    latest = panel[panel["month"] == latest_month].sort_values("ticker").head(80)
    if not latest.empty:
        parts.append(_sample_with_source(latest, "latest_month_unlabeled_or_partially_labeled"))

    # 2) Recent labeled months: include both top tail names and ordinary non-tail names.
    if "future_max_return_1_3m" in panel.columns:
        labeled = panel[panel["future_max_return_1_3m"].notna()].copy()
    else:
        labeled = pd.DataFrame()

    if not labeled.empty:
        recent_months = sorted(labeled["month"].dropna().unique())[-12:]
        recent = labeled[labeled["month"].isin(recent_months)]
        for month, g in recent.groupby("month", sort=True):
            top_tail = g.sort_values("future_max_return_1_3m", ascending=False).head(10)
            ordinary = g[g.get("label_top10_1_3m", 0).fillna(0).astype(int) == 0].sort_values("ticker").head(10)
            if not top_tail.empty:
                parts.append(_sample_with_source(top_tail, "recent_month_top_future_return"))
            if not ordinary.empty:
                parts.append(_sample_with_source(ordinary, "recent_month_non_top10_control"))

        # 3) Extreme historical positive examples for tail-event modeling.
        for label, source, n in [
            ("label_mega100_1_3m", "historical_mega100_examples", 120),
            ("label_boom50_top5_1_3m", "historical_boom50_top5_examples", 120),
            ("label_boom30_top10_1_3m", "historical_boom30_top10_examples", 120),
        ]:
            if label in labeled.columns:
                pos = labeled[labeled[label].fillna(0).astype(int) == 1]
                pos = pos.sort_values("future_max_return_1_3m", ascending=False).head(n)
                if not pos.empty:
                    parts.append(_sample_with_source(pos, source))

        # 4) Yearly top examples across the whole backtest period.
        yearly_parts = []
        for year, g in labeled.groupby(labeled["month"].dt.year):
            yearly_parts.append(g.sort_values("future_max_return_1_3m", ascending=False).head(8))
        if yearly_parts:
            parts.append(_sample_with_source(pd.concat(yearly_parts, ignore_index=True), "yearly_top_future_return_examples"))

    if parts:
        sample = pd.concat(parts, ignore_index=True)
        sample = sample.drop_duplicates(["month", "ticker"], keep="first")
        sample = sample.sort_values(["month", "sample_source", "ticker"]).reset_index(drop=True)
    else:
        sample = panel.head(200).copy()
        sample.insert(0, "sample_source", "fallback_first_rows")

    # Keep the file small enough for normal GitHub browsing.
    sample.head(1000).to_csv(MODEL_DESIGN_SAMPLE_FILE, index=False)


def _write_summaries(raw: pd.DataFrame, clean: pd.DataFrame, missing_report: pd.DataFrame, dropped: pd.DataFrame) -> None:
    labels = [c for c in LABEL_COLUMNS if c.startswith("label_") and c in raw.columns]
    rows = []
    for c in labels:
        valid_rows = raw["future_max_return_1_3m"].notna().sum()
        rows.append({
            "label": c,
            "positive_rows": int(raw[c].sum()),
            "valid_future_rows": int(valid_rows),
            "positive_rate": float(raw[c].sum() / valid_rows) if valid_rows else np.nan,
            "months_with_positive": int(raw.groupby("month")[c].sum().gt(0).sum()),
            "avg_positives_per_month": float(raw.groupby("month")[c].sum().mean()),
            "min_positives_per_month": int(raw.groupby("month")[c].sum().min()),
            "max_positives_per_month": int(raw.groupby("month")[c].sum().max()),
        })
    pd.DataFrame(rows).to_csv(LABEL_SUMMARY_FILE, index=False)

    by_month = raw.groupby("month").agg(
        rows=("ticker", "count"),
        tickers=("ticker", "nunique"),
        future_max_return_1_3m_mean=("future_max_return_1_3m", "mean"),
        future_max_return_1_3m_p90=("future_max_return_1_3m", lambda s: s.quantile(0.90)),
        future_max_return_1_3m_p95=("future_max_return_1_3m", lambda s: s.quantile(0.95)),
        future_max_return_1_3m_max=("future_max_return_1_3m", "max"),
    ).reset_index()
    for c in labels:
        by_month[c + "_count"] = raw.groupby("month")[c].sum().values
    by_month.to_csv(LABEL_BY_MONTH_FILE, index=False)

    summary = pd.DataFrame({
        "metric": [
            "raw_rows", "clean_rows", "raw_columns", "clean_columns", "raw_tickers", "clean_tickers", "months",
            "first_month", "last_month", "candidate_features", "kept_features", "dropped_features",
        ],
        "value": [
            len(raw), len(clean), len(raw.columns), len(clean.columns), raw["ticker"].nunique(), clean["ticker"].nunique(), clean["month"].nunique(),
            clean["month"].min(), clean["month"].max(), len(missing_report), int((~missing_report["dropped"]).sum()), len(dropped),
        ],
    })
    summary.to_csv(PANEL_SUMMARY_FILE, index=False)

    # Latest rows are useful for live prediction inspection, but their future labels
    # are usually blank. Keep this small file in GitHub for quick README display.
    latest = clean[clean["month"] == clean["month"].max()].copy().head(100)
    latest.to_csv(LATEST_PANEL_SAMPLE_FILE, index=False)

    # A richer, repository-friendly sample for model design and debugging.
    # It intentionally includes historical rows with realized future returns / labels,
    # plus a latest-month slice. The full panel remains an Actions artifact.
    _write_model_design_sample(clean)


def build_panel() -> pd.DataFrame:
    if not DAILY_PRICES_FILE.exists():
        raise FileNotFoundError(f"Missing {DAILY_PRICES_FILE}. Run python -m src.download_data first.")
    prices = pd.read_csv(DAILY_PRICES_FILE, parse_dates=["date"])
    daily = _prepare_daily(prices)
    month_end = _daily_month_end_features(daily).sort_values(["ticker", "month"])
    panel = month_end.copy()
    panel = _attach_source_features(panel)
    g = panel.groupby("ticker")

    # Monthly momentum levels.
    for k in [1, 2, 3, 4, 5, 6, 7, 9, 12]:
        panel[f"mom_{k}m"] = g["adj_close"].pct_change(k)

    # Single-month return path.
    panel["ret_lag_1m"] = panel["mom_1m"]
    for lag in range(2, 7):
        panel[f"ret_lag_{lag}m"] = g["ret_lag_1m"].shift(lag - 1)
    lag_cols = [f"ret_lag_{i}m" for i in range(1, 7)]
    panel["positive_month_ratio_6m"] = panel[lag_cols].gt(0).sum(axis=1) / panel[lag_cols].notna().sum(axis=1)
    panel["ret_lag_6m_mean"] = panel[lag_cols].mean(axis=1)
    panel["ret_lag_6m_std"] = panel[lag_cols].std(axis=1)
    panel["ret_lag_6m_min"] = panel[lag_cols].min(axis=1)
    panel["ret_lag_6m_max"] = panel[lag_cols].max(axis=1)

    # Core 4/5/6 month momentum shape.
    core_cols = ["mom_4m", "mom_5m", "mom_6m"]
    panel["core_mom_456_avg"] = panel[core_cols].mean(axis=1)
    panel["core_mom_456_min"] = panel[core_cols].min(axis=1)
    panel["core_mom_456_max"] = panel[core_cols].max(axis=1)
    panel["core_mom_456_std"] = panel[core_cols].std(axis=1)
    panel["mom_4m_vs_6m"] = panel["mom_4m"] - panel["mom_6m"]
    panel["mom_5m_vs_6m"] = panel["mom_5m"] - panel["mom_6m"]
    panel["mom_6m_first3m"] = g["adj_close"].shift(3) / g["adj_close"].shift(6) - 1
    panel["mom_6m_last3m"] = panel["adj_close"] / g["adj_close"].shift(3) - 1
    panel["mom_6m_acceleration"] = panel["mom_6m_last3m"] - panel["mom_6m_first3m"]
    panel["mom_3m_vs_6m"] = panel["mom_3m"] - panel["mom_6m"]

    # Moving average ratios.
    for w in [5, 10, 20, 30, 50, 100]:
        if f"ma{w}" in panel.columns:
            panel[f"price_ma{w}_ratio"] = panel["adj_close"] / panel[f"ma{w}"].replace(0, np.nan) - 1

    # Drawdown / pullback.
    for name in ["1m", "3m", "6m", "12m"]:
        if f"drawdown_{name}" in panel.columns:
            panel[f"drawdown_{name}_abs"] = -panel[f"drawdown_{name}"]
    panel["drawdown_change_1m"] = panel["drawdown_3m"] - g["drawdown_3m"].shift(1)
    panel["drawdown_change_3m"] = panel["drawdown_3m"] - g["drawdown_3m"].shift(3)

    # Volatility structure and risk-adjusted momentum.
    panel["volatility_1m_to_3m"] = panel["volatility_1m"] / panel["volatility_3m"].replace(0, np.nan)
    panel["volatility_1m_to_6m"] = panel["volatility_1m"] / panel["volatility_6m"].replace(0, np.nan)
    panel["volatility_3m_to_6m"] = panel["volatility_3m"] / panel["volatility_6m"].replace(0, np.nan)
    panel["return_vol_ratio_1m"] = panel["mom_1m"] / panel["volatility_1m"].replace(0, np.nan)
    panel["return_vol_ratio_3m"] = panel["mom_3m"] / panel["volatility_3m"].replace(0, np.nan)
    panel["return_vol_ratio_6m"] = panel["mom_6m"] / panel["volatility_6m"].replace(0, np.nan)

    # Volume features.
    panel["volume_change_1m"] = g["volume"].pct_change(1)
    panel["volume_change_3m"] = g["volume"].pct_change(3)
    panel["volume_ma3"] = g["volume"].transform(lambda s: s.rolling(3, min_periods=2).mean())
    panel["volume_ma6"] = g["volume"].transform(lambda s: s.rolling(6, min_periods=3).mean())
    panel["volume_ma12"] = g["volume"].transform(lambda s: s.rolling(12, min_periods=6).mean())
    panel["volume_ratio_3m"] = panel["volume"] / panel["volume_ma3"].replace(0, np.nan) - 1
    panel["volume_ma1_to_6m"] = panel["volume"] / panel["volume_ma6"].replace(0, np.nan)
    panel["volume_ma3_to_12m"] = panel["volume_ma3"] / panel["volume_ma12"].replace(0, np.nan)
    panel["dollar_volume_change_1m"] = g["dollar_volume"].pct_change(1)
    panel["dollar_volume_change_3m"] = g["dollar_volume"].pct_change(3)

    # Future returns and tail-event labels.
    for h in [1, 2, 3]:
        panel[f"future_return_{h}m"] = g["adj_close"].shift(-h) / panel["adj_close"] - 1
    panel["future_max_return_1_3m"] = panel[["future_return_1m", "future_return_2m", "future_return_3m"]].max(axis=1)
    panel = _add_tail_labels(panel)

    # Benchmark QQQ context.
    qqq = panel[panel["ticker"] == BENCHMARK].copy()
    qqq = qqq[["month", "mom_1m", "mom_3m", "mom_6m", "mom_12m"]].rename(columns={
        "mom_1m": "qqq_mom_1m",
        "mom_3m": "qqq_mom_3m",
        "mom_6m": "qqq_mom_6m",
        "mom_12m": "qqq_mom_12m",
    })
    panel = panel.merge(qqq, on="month", how="left")
    for k in [1, 3, 6, 12]:
        panel[f"rel_mom_{k}m_vs_qqq"] = panel[f"mom_{k}m"] - panel[f"qqq_mom_{k}m"]

    # Remove benchmark ETF row from stock-selection panel if present.
    panel = panel[panel["ticker"] != BENCHMARK]
    panel = panel[panel["month"] >= pd.Timestamp(FIRST_SAMPLE_MONTH)]
    panel = _compute_liquid_vol_score(panel)

    numeric_cols = panel.select_dtypes(include=[np.number]).columns
    panel[numeric_cols] = panel[numeric_cols].replace([np.inf, -np.inf], np.nan)
    panel[numeric_cols] = panel[numeric_cols].mask(panel[numeric_cols].abs() > 1e12)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = panel.sort_values(["month", "ticker"]).reset_index(drop=True)
    raw.to_csv(RAW_PANEL_FILE, index=False)

    candidate_features = all_declared_features()
    clean, kept_features, missing_report, dropped = _filter_missing(raw, candidate_features)
    clean = clean.sort_values(["month", "ticker"]).reset_index(drop=True)
    clean.to_csv(CLEAN_PANEL_FILE, index=False)

    # A committed modeling reference slice. The full panel is uploaded as an artifact,
    # while this first-20k-row sample stays small enough for GitHub and future model design.
    clean.head(20000).to_csv(PANEL_HEAD_20000_FILE, index=False)

    missing_report.to_csv(FEATURE_MISSING_REPORT_FILE, index=False)
    dropped.to_csv(DROPPED_FEATURES_FILE, index=False)
    _write_manifest(clean, kept_features, dropped)
    _write_summaries(raw, clean, missing_report, dropped)

    print(f"Saved raw panel: {RAW_PANEL_FILE}")
    print(f"Saved clean panel: {CLEAN_PANEL_FILE}")
    print(f"Rows={len(clean):,}, columns={len(clean.columns):,}, tickers={clean['ticker'].nunique():,}, months={clean['month'].nunique():,}")
    print(f"Kept features={len(kept_features)}, dropped high-missing features={len(dropped)}")
    return clean


if __name__ == "__main__":
    build_panel()
