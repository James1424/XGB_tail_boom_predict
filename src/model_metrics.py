from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .model_config import FUTURE_RETURN_COLS, TARGET_LABEL, TOP_KS


def safe_auc(y_true, score, kind="pr") -> float:
    y = pd.Series(y_true).dropna()
    if y.nunique() < 2:
        return float("nan")
    idx = y.index
    s = pd.Series(score).loc[idx]
    try:
        if kind == "roc":
            return float(roc_auc_score(y, s))
        return float(average_precision_score(y, s))
    except Exception:
        return float("nan")


def topk_metrics(df: pd.DataFrame, score_col: str, label_col: str = TARGET_LABEL, prefix: str = "") -> dict:
    rows = df.dropna(subset=[score_col]).copy()
    if rows.empty:
        return {}
    out = {}
    valid_label = rows.dropna(subset=[label_col]) if label_col in rows.columns else rows
    if label_col in rows.columns and not valid_label.empty:
        out[f"{prefix}pr_auc"] = safe_auc(valid_label[label_col], valid_label[score_col], "pr")
        out[f"{prefix}roc_auc"] = safe_auc(valid_label[label_col], valid_label[score_col], "roc")
    for k in TOP_KS:
        top = rows.sort_values(["month", score_col], ascending=[True, False]).groupby("month", group_keys=False).head(k)
        out[f"{prefix}top{k}_rows"] = int(len(top))
        if label_col in top.columns:
            out[f"{prefix}precision_at_top{k}"] = float(top[label_col].mean()) if len(top) else np.nan
        for col in FUTURE_RETURN_COLS:
            if col in top.columns:
                out[f"{prefix}top{k}_avg_{col}"] = float(top[col].mean())
        if "future_max_return_1_3m" in top.columns:
            fm = top["future_max_return_1_3m"]
            out[f"{prefix}top{k}_hit30_rate"] = float((fm >= 0.30).mean())
            out[f"{prefix}top{k}_hit40_rate"] = float((fm >= 0.40).mean())
            out[f"{prefix}top{k}_hit50_rate"] = float((fm >= 0.50).mean())
            out[f"{prefix}top{k}_hit100_rate"] = float((fm >= 1.00).mean())
            monthly = top.groupby("month")["future_max_return_1_3m"]
            out[f"{prefix}monthly_any_top{k}_hit30_rate"] = float(monthly.apply(lambda s: (s >= 0.30).any()).mean())
            out[f"{prefix}monthly_any_top{k}_hit50_rate"] = float(monthly.apply(lambda s: (s >= 0.50).any()).mean())
            out[f"{prefix}monthly_any_top{k}_hit100_rate"] = float(monthly.apply(lambda s: (s >= 1.00).any()).mean())
    return out


def monthly_top_table(df: pd.DataFrame, score_col: str, k: int = 10) -> pd.DataFrame:
    keep = ["month", "ticker", score_col]
    for c in ["future_return_1m", "future_return_2m", "future_return_3m", "future_max_return_1_3m", TARGET_LABEL,
              "label_boom40_top10_1_3m", "label_boom50_top5_1_3m", "label_mega100_1_3m"]:
        if c in df.columns:
            keep.append(c)
    out = df.dropna(subset=[score_col]).sort_values(["month", score_col], ascending=[True, False]).groupby("month", group_keys=False).head(k)[keep].copy()
    out["rank"] = out.groupby("month")[score_col].rank(method="first", ascending=False).astype(int)
    return out.sort_values(["month", "rank"])


def strategy_row(name: str, df: pd.DataFrame, score_col: str, split: str = "test") -> dict:
    m = topk_metrics(df, score_col, TARGET_LABEL, prefix="")
    row = {"strategy": name, "split": split}
    row.update(m)
    return row
