from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .model_config import (
    ABLATION_GROUPS,
    BASE_MODEL_PARAMS,
    FEATURE_LIST_FILE,
    MAIN_MODEL_FILE,
    MAIN_SEEDS,
    MODEL_DIR,
    OUTPUT_DIR,
    OUTPUT_FILES,
    ROUND_MODEL_PARAMS,
    ROUND_SEEDS,
    TARGET_LABEL,
    TEST_START,
    TRAIN_END,
    VALID_END,
    VALID_START,
)
from .model_features import columns_matching, infer_feature_columns, load_panel
from .model_metrics import monthly_top_table, safe_auc, strategy_row, topk_metrics


def split_panel(df: pd.DataFrame):
    train = df[(df["month"] <= pd.Timestamp(TRAIN_END)) & df[TARGET_LABEL].notna()].copy()
    valid = df[(df["month"] >= pd.Timestamp(VALID_START)) & (df["month"] <= pd.Timestamp(VALID_END)) & df[TARGET_LABEL].notna()].copy()
    test = df[(df["month"] >= pd.Timestamp(TEST_START)) & df[TARGET_LABEL].notna()].copy()
    latest_month = df["month"].max()
    latest = df[df["month"] == latest_month].copy()
    return train, valid, test, latest


def make_reference_downweighted_weights(df: pd.DataFrame, target_col: str = TARGET_LABEL) -> np.ndarray:
    y = df[target_col].fillna(0).astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    base_pos = max(1.0, neg / max(pos, 1))
    w = np.where(y == 1, base_pos, 0.55).astype(float)

    # Downweight easy/reference negatives: weak or negative future return. These rows are useful,
    # but they should not dominate the tail-event learning objective.
    if "future_max_return_1_3m" in df.columns:
        fm = df["future_max_return_1_3m"].fillna(0)
        w[(y == 0) & (fm <= 0)] *= 0.55
        w[(y == 0) & (fm > 0.20)] *= 1.25
    # Strengthen progressively rarer right-tail labels when available.
    boost_cols = [
        ("label_boom40_top10_1_3m", 1.25),
        ("label_boom50_top5_1_3m", 1.50),
        ("label_mega100_1_3m", 2.00),
    ]
    for col, mult in boost_cols:
        if col in df.columns:
            w[df[col].fillna(0).astype(int) == 1] *= mult
    return w


def fit_xgb(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], seed: int, params: dict | None = None) -> XGBClassifier:
    p = dict(BASE_MODEL_PARAMS if params is None else params)
    p["random_state"] = seed
    model = XGBClassifier(**p)
    X_train = train[features].replace([np.inf, -np.inf], np.nan)
    y_train = train[TARGET_LABEL].astype(int)
    w_train = make_reference_downweighted_weights(train)
    X_valid = valid[features].replace([np.inf, -np.inf], np.nan)
    y_valid = valid[TARGET_LABEL].astype(int)
    model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model


def predict(model: XGBClassifier, df: pd.DataFrame, features: list[str], score_col: str) -> pd.DataFrame:
    out = df.copy()
    out[score_col] = model.predict_proba(out[features].replace([np.inf, -np.inf], np.nan))[:, 1]
    return out


def train_main(train, valid, test, latest, features):
    model = fit_xgb(train, valid, features, seed=42, params=BASE_MODEL_PARAMS)
    pred_test = predict(model, test, features, "main_score")
    pred_latest = predict(model, latest, features, "main_score")
    metrics = topk_metrics(pred_test, "main_score", TARGET_LABEL)
    metrics.update({
        "model": "reference_downweighted_xgb_classifier",
        "target": TARGET_LABEL,
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "feature_count": int(len(features)),
    })
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MAIN_MODEL_FILE))
    FEATURE_LIST_FILE.write_text("\n".join(features), encoding="utf-8")
    imp = pd.DataFrame({"feature": features, "importance_gain": model.feature_importances_}).sort_values("importance_gain", ascending=False)
    imp.to_csv(OUTPUT_FILES["feature_importance"], index=False)
    return model, pred_test, pred_latest, metrics


def baseline_comparison(test_pred: pd.DataFrame) -> pd.DataFrame:
    rows = [strategy_row("Reference-downweighted XGB main model", test_pred, "main_score")]
    baselines = {
        "Baseline: 6M momentum": "mom_6m",
        "Baseline: 3M momentum": "mom_3m",
        "Baseline: 12M momentum": "mom_12m",
        "Baseline: relative 6M vs QQQ": "rel_mom_6m_vs_qqq",
        "Baseline: core momentum avg 4/5/6M": "core_mom_456_avg",
        "Baseline: liquid volatility score": "liquid_vol_score",
        "Baseline: dollar volume 3M": "log_avg_dollar_volume_3m",
    }
    for name, col in baselines.items():
        if col in test_pred.columns:
            tmp = test_pred.copy()
            tmp[col] = tmp[col].fillna(tmp[col].median())
            rows.append(strategy_row(name, tmp, col))
    return pd.DataFrame(rows)


def five_seed_stability(train, valid, test, latest, features):
    rows = []
    latest_scores = latest[["month", "ticker"]].copy()
    test_scores = test[["month", "ticker"]].copy()
    for seed in MAIN_SEEDS:
        m = fit_xgb(train, valid, features, seed=seed, params=BASE_MODEL_PARAMS)
        score_col = f"seed_{seed}_score"
        pt = predict(m, test, features, score_col)
        pl = predict(m, latest, features, score_col)
        row = {"seed": seed}
        row.update(topk_metrics(pt, score_col, TARGET_LABEL))
        rows.append(row)
        latest_scores = latest_scores.merge(pl[["month", "ticker", score_col]], on=["month", "ticker"], how="left")
        test_scores = test_scores.merge(pt[["month", "ticker", score_col]], on=["month", "ticker"], how="left")
    seed_cols = [c for c in latest_scores.columns if c.endswith("_score")]
    latest_scores["five_seed_avg_score"] = latest_scores[seed_cols].mean(axis=1)
    latest_scores["five_seed_score_std"] = latest_scores[seed_cols].std(axis=1)
    return pd.DataFrame(rows), latest_scores, test_scores


def run_100_rounds(train, valid, test, latest, features, rounds: int):
    rows = []
    candidate_records = []
    seeds = ROUND_SEEDS[:rounds]
    all_latest = latest[["month", "ticker"]].copy()
    for seed in seeds:
        rng = np.random.default_rng(seed)
        train_months = np.array(sorted(train["month"].unique()))
        keep_months = rng.choice(train_months, size=max(3, int(len(train_months) * 0.85)), replace=False)
        tr = train[train["month"].isin(keep_months)].copy()
        feat_keep = [f for f in features if rng.random() > 0.08]
        if len(feat_keep) < max(10, len(features) // 2):
            feat_keep = features
        m = fit_xgb(tr, valid, feat_keep, seed=seed, params=ROUND_MODEL_PARAMS)
        score_col = f"round_{seed}_score"
        pt = predict(m, test, feat_keep, score_col)
        pl = predict(m, latest, feat_keep, score_col)
        row = {"round": seed, "train_months": len(keep_months), "feature_count": len(feat_keep)}
        row.update(topk_metrics(pt, score_col, TARGET_LABEL))
        rows.append(row)
        top_latest = pl.sort_values(score_col, ascending=False).head(10).copy()
        top_latest["round"] = seed
        top_latest["rank"] = range(1, len(top_latest) + 1)
        candidate_records.append(top_latest[["round", "rank", "month", "ticker", score_col]])
        all_latest = all_latest.merge(pl[["month", "ticker", score_col]], on=["month", "ticker"], how="left")
    report = pd.DataFrame(rows)
    cand = pd.concat(candidate_records, ignore_index=True) if candidate_records else pd.DataFrame()
    if not cand.empty:
        freq = cand.groupby("ticker").agg(
            selected_rounds=("round", "nunique"),
            avg_rank=("rank", "mean"),
            best_rank=("rank", "min"),
        ).reset_index().sort_values(["selected_rounds", "avg_rank"], ascending=[False, True])
    else:
        freq = pd.DataFrame()
    return report, freq, all_latest


def ablation_summary(train, valid, test, features, reference_top3: float | None = None):
    rows = []
    for group, patterns in ABLATION_GROUPS.items():
        drop_cols = columns_matching(features, patterns)
        if not drop_cols:
            continue
        keep = [f for f in features if f not in drop_cols]
        if len(keep) < 10:
            continue
        m = fit_xgb(train, valid, keep, seed=42, params=BASE_MODEL_PARAMS)
        pt = predict(m, test, keep, f"abl_{group}_score")
        met = topk_metrics(pt, f"abl_{group}_score", TARGET_LABEL)
        row = {
            "ablation": f"drop_{group}",
            "dropped_features": len(drop_cols),
            "kept_features": len(keep),
        }
        row.update(met)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and reference_top3 is not None and "top3_avg_future_max_return_1_3m" in out.columns:
        out["delta_top3_future_max_vs_main"] = out["top3_avg_future_max_return_1_3m"] - reference_top3
        out = out.sort_values("delta_top3_future_max_vs_main")
    return out


def make_latest_candidates(latest: pd.DataFrame, main_latest: pd.DataFrame, five_seed_latest: pd.DataFrame, rounds_latest: pd.DataFrame | None = None) -> pd.DataFrame:
    out = main_latest.copy()
    out = out.merge(five_seed_latest[["month", "ticker", "five_seed_avg_score", "five_seed_score_std"]], on=["month", "ticker"], how="left")
    if rounds_latest is not None and not rounds_latest.empty:
        round_score_cols = [c for c in rounds_latest.columns if c.startswith("round_") and c.endswith("_score")]
        rounds_latest = rounds_latest.copy()
        rounds_latest["rounds_100_avg_score"] = rounds_latest[round_score_cols].mean(axis=1)
        rounds_latest["rounds_100_score_std"] = rounds_latest[round_score_cols].std(axis=1)
        out = out.merge(rounds_latest[["month", "ticker", "rounds_100_avg_score", "rounds_100_score_std"]], on=["month", "ticker"], how="left")
    score_cols = [c for c in ["main_score", "five_seed_avg_score", "rounds_100_avg_score"] if c in out.columns]
    out["ensemble_score"] = out[score_cols].mean(axis=1)
    keep_cols = ["month", "ticker", "ensemble_score", "main_score", "five_seed_avg_score", "five_seed_score_std", "rounds_100_avg_score", "rounds_100_score_std"]
    context_cols = ["mom_6m", "mom_3m", "rel_mom_6m_vs_qqq", "liquid_vol_score", "avg_dollar_volume_3m", "large_move_freq_3m", "source_count", "source_weight_sum", "theme_count"]
    keep_cols += [c for c in context_cols if c in out.columns and c not in keep_cols]
    keep_cols = [c for c in keep_cols if c in out.columns]
    final = out.sort_values("ensemble_score", ascending=False).head(30)[keep_cols].copy()
    final["rank"] = range(1, len(final) + 1)
    return final[["rank"] + [c for c in final.columns if c != "rank"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=100, help="Number of randomized rounds for the 100-round report.")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-rounds", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_panel()
    if TARGET_LABEL not in df.columns:
        raise ValueError(f"Missing target label {TARGET_LABEL}")
    features = infer_feature_columns(df)
    train, valid, test, latest = split_panel(df)
    if train.empty or valid.empty or test.empty:
        raise ValueError("Train/valid/test split has empty segment. Check month coverage and labels.")

    main_model, pred_test, pred_latest, main_metrics = train_main(train, valid, test, latest, features)
    pd.DataFrame([main_metrics]).to_csv(OUTPUT_FILES["main_result"], index=False)

    baseline = baseline_comparison(pred_test)
    baseline.to_csv(OUTPUT_FILES["strategy_baseline"], index=False)

    five_report, five_latest, five_test_scores = five_seed_stability(train, valid, test, latest, features)
    five_report.to_csv(OUTPUT_FILES["five_seed"], index=False)

    rounds_report = pd.DataFrame()
    rounds_freq = pd.DataFrame()
    rounds_latest = pd.DataFrame()
    if not args.skip_rounds:
        rounds_report, rounds_freq, rounds_latest = run_100_rounds(train, valid, test, latest, features, rounds=args.rounds)
        rounds_report.to_csv(OUTPUT_FILES["rounds_100"], index=False)
        rounds_freq.to_csv(OUTPUT_FILES["rounds_frequency"], index=False)
    else:
        pd.DataFrame().to_csv(OUTPUT_FILES["rounds_100"], index=False)
        pd.DataFrame().to_csv(OUTPUT_FILES["rounds_frequency"], index=False)

    if not args.skip_ablation:
        ref_val = main_metrics.get("top3_avg_future_max_return_1_3m")
        abl = ablation_summary(train, valid, test, features, reference_top3=ref_val)
        abl.to_csv(OUTPUT_FILES["ablation"], index=False)
    else:
        pd.DataFrame().to_csv(OUTPUT_FILES["ablation"], index=False)

    latest_candidates = make_latest_candidates(latest, pred_latest, five_latest, rounds_latest)
    latest_candidates.to_csv(OUTPUT_FILES["latest_live"], index=False)

    top_monthly = monthly_top_table(pred_test, "main_score", k=10)
    top_monthly.to_csv(OUTPUT_FILES["monthly_top"], index=False)
    pred_test.to_csv(OUTPUT_FILES["full_predictions"], index=False)

    metrics_json = {
        "target_label": TARGET_LABEL,
        "features": features,
        "main_metrics": main_metrics,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "test_rows": len(test),
        "latest_month": str(latest["month"].max().date()),
        "rounds_requested": args.rounds if not args.skip_rounds else 0,
    }
    OUTPUT_FILES["metrics_json"].write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    print("Training complete. Key outputs written to outputs/ and models/.")


if __name__ == "__main__":
    main()
