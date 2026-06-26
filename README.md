# Integrated ETF Tail Boom Prediction Project

This repository combines two stages in one GitHub project:

1. **Panel construction**: build an ETF/index-holdings monthly panel for right-tail stock-event detection.
2. **Model training**: train an XGBoost right-tail boom classifier on the generated panel and publish model reports.

The full panel can be larger than GitHub's 100 MB file limit, so the workflow keeps large files as **GitHub Actions artifacts** while committing smaller reports and model summaries to the repository.

## Main target

The primary training label is:

```text
label_boom30_top10_1_3m
```

A positive row means that the stock's future 1–3 month max return is both:

- in the monthly top 10%; and
- at least +30%.

## Run locally

```bash
pip install -r requirements.txt
python run_all.py
```

Step by step:

```bash
python -m src.get_holdings_universe
python -m src.download_data
python -m src.build_panel
python -m src.train_model --rounds 100
python -m src.update_model_readme
```

For a fast smoke test:

```bash
python -m src.train_model --rounds 5 --skip-ablation
python -m src.update_model_readme
```

## GitHub Actions workflow

Use:

```text
Actions -> Build Panel and Train Tail Boom Model -> Run workflow
```

The workflow does this in one run:

```text
Build ETF/index universe
→ Download daily prices
→ Build full monthly panel
→ Train XGBoost right-tail boom model
→ Upload full panel as artifact
→ Upload full predictions as artifact
→ Commit README, small reports, model JSON, and selected features
```

## Artifact policy

Large generated files are not committed to the repository:

```text
outputs/raw_monthly_panel.csv
outputs/clean_monthly_panel.csv
data/daily_prices.csv.gz
outputs/full_predictions.csv
```

They are uploaded as Actions artifacts and retained for 14 days.

Small reports are committed:

```text
outputs/panel_summary.csv
outputs/label_summary.csv
outputs/reference_downweighted_main_model_result.csv
outputs/latest_live_boom_candidates.csv
outputs/strategy_baseline_comparison.csv
outputs/ablation_ranked_summary.csv
outputs/rounds_100_report.csv
outputs/five_seed_training_stability.csv
outputs/feature_importance.csv
models/xgb_tail_event_classifier.json
models/selected_features.txt
```

## Leakage rule

Never use these columns as model inputs:

```text
future_return_1m
future_return_2m
future_return_3m
future_max_return_1_3m
future_max_return_1_3m_pct_rank
monthly_top10_threshold_1_3m
monthly_top5_threshold_1_3m
label_*
```

The training code automatically excludes future-return and label columns.

## Report sections generated after training

- 100-rounds report
- Latest live boom candidates
- Strategy and baseline comparison
- Ablation ranked summary
- Reference-downweighted main model result
- Five-seed training stability
