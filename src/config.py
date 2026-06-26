from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SOURCE_ETFS_FILE = DATA_DIR / "source_etfs.csv"
MANUAL_TICKERS_FILE = DATA_DIR / "manual_tickers.csv"
SEED_UNIVERSE_FILE = DATA_DIR / "seed_universe.csv"
DAILY_PRICES_FILE = DATA_DIR / "daily_prices.csv.gz"

RAW_PANEL_FILE = OUTPUT_DIR / "raw_monthly_panel.csv"
CLEAN_PANEL_FILE = OUTPUT_DIR / "clean_monthly_panel.csv"
FEATURE_MISSING_REPORT_FILE = OUTPUT_DIR / "feature_missing_report.csv"
DROPPED_FEATURES_FILE = OUTPUT_DIR / "dropped_features.csv"
FEATURE_MANIFEST_FILE = OUTPUT_DIR / "feature_manifest.txt"
PANEL_SUMMARY_FILE = OUTPUT_DIR / "panel_summary.csv"
LABEL_SUMMARY_FILE = OUTPUT_DIR / "label_summary.csv"
LABEL_BY_MONTH_FILE = OUTPUT_DIR / "label_distribution_by_month.csv"
LATEST_PANEL_SAMPLE_FILE = OUTPUT_DIR / "latest_panel_sample.csv"
MODEL_DESIGN_SAMPLE_FILE = OUTPUT_DIR / "model_design_panel_sample.csv"
PANEL_HEAD_20000_FILE = OUTPUT_DIR / "panel_head_20000.csv"
README_FILE = PROJECT_ROOT / "README.md"

START_DATE = "2014-01-01"
FIRST_SAMPLE_MONTH = "2016-01-31"
BENCHMARK = "QQQ"

# Missingness filter for model features. Future/label/meta columns are never model inputs.
MAX_FEATURE_MISSING_RATE = 0.35
MIN_FEATURE_NON_NULL_ROWS = 1000

# Tail labels.
TAIL_TOP10_Q = 0.90
TAIL_TOP5_Q = 0.95
BOOM30 = 0.30
BOOM40 = 0.40
BOOM50 = 0.50
MEGA100 = 1.00

# Large move frequency thresholds based on daily returns.
LARGE_MOVE_ABS_THRESHOLD = 0.03
UP_BIG_MOVE_THRESHOLD = 0.05
DOWN_BIG_MOVE_THRESHOLD = -0.05

# Optional yfinance batching.
DOWNLOAD_CHUNK_SIZE = 80
DOWNLOAD_SLEEP_SECONDS = 1.0
