import pandas as pd

from .config import (
    README_FILE, SOURCE_ETFS_FILE, SEED_UNIVERSE_FILE, RAW_PANEL_FILE, CLEAN_PANEL_FILE,
    FEATURE_MISSING_REPORT_FILE, DROPPED_FEATURES_FILE, PANEL_SUMMARY_FILE,
    LABEL_SUMMARY_FILE, LABEL_BY_MONTH_FILE, LATEST_PANEL_SAMPLE_FILE, MODEL_DESIGN_SAMPLE_FILE, PANEL_HEAD_20000_FILE,
)
from .feature_groups import LABEL_COLUMNS


def fmt_pct(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return str(x)


def fmt_num(x):
    if pd.isna(x):
        return ""
    try:
        f = float(x)
        if abs(f) >= 1_000_000:
            return f"{f:,.0f}"
        if abs(f) >= 1_000:
            return f"{f:,.2f}"
        return f"{f:.4f}"
    except Exception:
        return str(x)


def table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No data available._"
    return df.to_markdown(index=False)


def format_pct_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(fmt_pct)
    return out


def main() -> None:
    lines = []
    lines.append("# ETF / Index Tail-Event Panel Builder")
    lines.append("")
    lines.append("This repository builds a monthly machine-learning panel for detecting right-tail stock events: stocks that may enter an explosive 1–3 month return regime.")
    lines.append("")
    lines.append("It uses ETF / index holdings as the seed universe rather than copying the full common-stock universe. The panel includes the dense feature set from the previous XGBoost boom project, plus ETF-source features, liquidity/size proxies, volatility amplitude/frequency features, future-return targets, and multiple tail-event labels.")
    lines.append("")
    lines.append("## Run")
    lines.append("```bash\npip install -r requirements.txt\npython run_all.py\n```")
    lines.append("")
    lines.append("Step by step:")
    lines.append("```bash\npython -m src.get_holdings_universe\npython -m src.download_data\npython -m src.build_panel\npython -m src.update_readme\n```")
    lines.append("")

    lines.append("## Important leakage rule")
    lines.append("The future-return and label columns are generated for training targets and evaluation only. They must not be used as model input features.")
    lines.append("")
    lines.append("Target / label columns include:")
    for c in LABEL_COLUMNS:
        lines.append(f"- `{c}`")
    lines.append("")

    lines.append("## Source ETF / index list")
    if SOURCE_ETFS_FILE.exists():
        src = pd.read_csv(SOURCE_ETFS_FILE)
        lines.append(table(src))
    else:
        lines.append("_No source ETF file found._")
    lines.append("")

    if SEED_UNIVERSE_FILE.exists():
        seed = pd.read_csv(SEED_UNIVERSE_FILE)
        lines.append("## Seed universe summary")
        seed_summary = pd.DataFrame({
            "metric": ["tickers", "avg_source_count", "max_source_count", "avg_source_weight_sum", "max_source_weight_sum"],
            "value": [
                seed["ticker"].nunique(), seed.get("source_count", pd.Series(dtype=float)).mean(), seed.get("source_count", pd.Series(dtype=float)).max(),
                seed.get("source_weight_sum", pd.Series(dtype=float)).mean(), seed.get("source_weight_sum", pd.Series(dtype=float)).max(),
            ],
        })
        seed_summary["value"] = seed_summary["value"].map(fmt_num)
        lines.append(table(seed_summary))
        lines.append("")
        show_cols = [c for c in ["ticker", "source_count", "source_weight_sum", "theme_count", "sources", "categories"] if c in seed.columns]
        lines.append("### Highest source-score seed tickers")
        lines.append(table(seed[show_cols].head(30)))
        lines.append("")

    if PANEL_SUMMARY_FILE.exists():
        ps = pd.read_csv(PANEL_SUMMARY_FILE)
        lines.append("## Panel summary")
        lines.append(table(ps))
        lines.append("")

    if LABEL_SUMMARY_FILE.exists():
        ls = pd.read_csv(LABEL_SUMMARY_FILE)
        lines.append("## Tail-label summary")
        ls = format_pct_cols(ls, ["positive_rate"])
        for c in ["avg_positives_per_month"]:
            if c in ls.columns:
                ls[c] = ls[c].map(fmt_num)
        lines.append(table(ls))
        lines.append("")

    if LABEL_BY_MONTH_FILE.exists():
        bm = pd.read_csv(LABEL_BY_MONTH_FILE, parse_dates=["month"]).tail(18)
        lines.append("## Recent monthly label distribution")
        bm["month"] = bm["month"].dt.strftime("%Y-%m-%d")
        for c in ["future_max_return_1_3m_mean", "future_max_return_1_3m_p90", "future_max_return_1_3m_p95", "future_max_return_1_3m_max"]:
            if c in bm.columns:
                bm[c] = bm[c].map(fmt_pct)
        lines.append(table(bm))
        lines.append("")

    if FEATURE_MISSING_REPORT_FILE.exists():
        miss = pd.read_csv(FEATURE_MISSING_REPORT_FILE)
        lines.append("## Feature missingness report")
        lines.append(f"A feature is dropped if `missing_rate > 35%` or `non_null_rows < 1000`.")
        lines.append("")
        show = miss.sort_values("missing_rate", ascending=False).head(30).copy()
        for c in ["missing_rate", "non_null_rate"]:
            if c in show.columns:
                show[c] = show[c].map(fmt_pct)
        lines.append("### Highest-missing candidate features")
        lines.append(table(show))
        lines.append("")

    if DROPPED_FEATURES_FILE.exists():
        dropped = pd.read_csv(DROPPED_FEATURES_FILE)
        lines.append("### Dropped high-missing features")
        if dropped.empty:
            lines.append("_No features dropped by missingness filter._")
        else:
            show = dropped.copy()
            for c in ["missing_rate", "non_null_rate"]:
                if c in show.columns:
                    show[c] = show[c].map(fmt_pct)
            lines.append(table(show))
        lines.append("")

    if PANEL_HEAD_20000_FILE.exists():
        try:
            head_sample = pd.read_csv(PANEL_HEAD_20000_FILE, nrows=0)
            lines.append("## Committed first-20k panel slice")
            lines.append("`outputs/panel_head_20000.csv` contains the first 20,000 rows of the cleaned panel. It is committed for quick inspection and future model-design reference; the full panel remains in the GitHub Actions artifact.")
            lines.append(f"Columns in slice: {len(head_sample.columns):,}")
            lines.append("")
        except Exception:
            pass

    if MODEL_DESIGN_SAMPLE_FILE.exists():
        design = pd.read_csv(MODEL_DESIGN_SAMPLE_FILE, parse_dates=["month"]).head(40)
        lines.append("## Model-design panel sample")
        lines.append("This small committed sample contains historical labeled rows and recent examples for model-design inspection. The full panel is uploaded as a GitHub Actions artifact.")
        keep = [
            "sample_source", "month", "ticker", "adj_close", "mom_3m", "mom_6m", "core_mom_456_avg",
            "avg_dollar_volume_3m", "large_move_freq_6m", "up_big_move_freq_6m",
            "liquid_vol_score", "future_max_return_1_3m", "label_top10_1_3m",
            "label_boom30_top10_1_3m", "label_boom50_top5_1_3m", "label_mega100_1_3m",
        ]
        keep = [c for c in keep if c in design.columns]
        show = design[keep].copy()
        if "month" in show.columns:
            show["month"] = show["month"].dt.strftime("%Y-%m-%d")
        for c in show.columns:
            if c not in ["sample_source", "month", "ticker"]:
                if c.startswith(("mom_", "core_", "large_", "up_", "liquid_", "future_")):
                    show[c] = show[c].map(fmt_pct)
                elif "dollar" in c:
                    show[c] = show[c].map(fmt_num)
        lines.append(table(show))
        lines.append("")

    if LATEST_PANEL_SAMPLE_FILE.exists():
        latest = pd.read_csv(LATEST_PANEL_SAMPLE_FILE, parse_dates=["month"]).head(30)
        lines.append("## Latest clean panel sample")
        keep = [
            "month", "ticker", "adj_close", "mom_3m", "mom_6m", "core_mom_456_avg",
            "avg_dollar_volume_3m", "large_move_freq_6m", "up_big_move_freq_6m",
            "liquid_vol_score", "future_max_return_1_3m", "label_boom30_top10_1_3m", "label_boom50_top5_1_3m",
        ]
        keep = [c for c in keep if c in latest.columns]
        show = latest[keep].copy()
        if "month" in show.columns:
            show["month"] = show["month"].dt.strftime("%Y-%m-%d")
        for c in show.columns:
            if c not in ["month", "ticker"] and c.startswith(("mom_", "core_", "avg_", "large_", "up_", "liquid_", "future_")):
                show[c] = show[c].map(fmt_pct if "dollar" not in c else fmt_num)
        lines.append(table(show))
        lines.append("")

    lines.append("## Output files")
    lines.append("- `data/seed_universe.csv`: ETF/index holdings universe with source metadata.")
    lines.append("- `data/daily_prices.csv.gz`: downloaded historical daily OHLCV data, ignored by Git because it can be large.")
    lines.append("- `outputs/raw_monthly_panel.csv`: full monthly panel before missingness feature deletion.")
    lines.append("- `outputs/clean_monthly_panel.csv`: training-ready panel after high-missing feature deletion.")
    lines.append("- `outputs/feature_missing_report.csv`: missingness statistics for candidate model features.")
    lines.append("- `outputs/dropped_features.csv`: removed high-missing features.")
    lines.append("- `outputs/label_summary.csv`: right-tail label distribution summary.")
    lines.append("- `outputs/label_distribution_by_month.csv`: monthly tail-label distribution.")
    lines.append("- `outputs/feature_manifest.txt`: final kept features and target columns.")
    lines.append("")
    lines.append("## Notes")
    lines.append("- ETF holdings URLs can change. If a source fails, check `data/holding_source_failures.csv` and add tickers to `data/manual_tickers.csv`.")
    lines.append("- This project builds the panel only. Model training should read `outputs/clean_monthly_panel.csv`, choose a target label, and exclude all future/label columns from `X_train`.")

    README_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Updated {README_FILE}")


if __name__ == "__main__":
    main()
