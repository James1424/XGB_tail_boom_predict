import argparse
import time
from typing import Iterable

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from .config import SEED_UNIVERSE_FILE, DAILY_PRICES_FILE, DATA_DIR, START_DATE, DOWNLOAD_CHUNK_SIZE, DOWNLOAD_SLEEP_SECONDS, BENCHMARK


def chunks(xs: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(xs), n):
        yield xs[i:i+n]


def _normalize_download(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                continue
            one = raw[t].copy()
            if one.empty:
                continue
            one["ticker"] = t
            one = one.reset_index().rename(columns={"Date": "date"})
            frames.append(one)
    else:
        one = raw.copy().reset_index().rename(columns={"Date": "date"})
        one["ticker"] = tickers[0] if tickers else ""
        frames.append(one)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    # yfinance auto_adjust=True returns close as adjusted close. Keep both names compatible.
    if "close" in out.columns and "adj_close" not in out.columns:
        out["adj_close"] = out["close"]
    keep = [c for c in ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"] if c in out.columns]
    out = out[keep]
    out = out.dropna(subset=["date", "ticker", "adj_close"])
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["ticker"] = out["ticker"].astype(str).str.upper()
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=DOWNLOAD_CHUNK_SIZE)
    parser.add_argument("--sleep", type=float, default=DOWNLOAD_SLEEP_SECONDS)
    parser.add_argument("--max-tickers", type=int, default=0, help="Optional cap after ETF universe is built. 0 means all seed tickers.")
    args = parser.parse_args(argv)

    if not SEED_UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"Missing {SEED_UNIVERSE_FILE}; run python -m src.get_holdings_universe first")

    uni = pd.read_csv(SEED_UNIVERSE_FILE)
    tickers = uni["ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    if BENCHMARK not in tickers:
        tickers.append(BENCHMARK)
    if args.max_tickers and args.max_tickers > 0:
        # This cap is only for debugging. The universe is already sorted by ETF source quality, not alphabetically.
        tickers = tickers[:args.max_tickers]
        if BENCHMARK not in tickers:
            tickers.append(BENCHMARK)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_frames = []
    failures = []
    for batch in tqdm(list(chunks(tickers, args.chunk_size)), desc="Downloading yfinance batches"):
        try:
            raw = yf.download(
                tickers=batch,
                start=START_DATE,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            norm = _normalize_download(raw, batch)
            got = set(norm["ticker"].unique()) if not norm.empty else set()
            for t in batch:
                if t not in got:
                    failures.append({"ticker": t, "reason": "no_rows_in_batch"})
            if not norm.empty:
                all_frames.append(norm)
        except Exception as e:
            print(f"Download failed for batch {batch[:5]}...: {e}")
            failures.extend({"ticker": t, "reason": str(e)[:200]} for t in batch)
        time.sleep(args.sleep)

    if not all_frames:
        raise RuntimeError("No price data downloaded.")

    daily = pd.concat(all_frames, ignore_index=True)
    daily = daily.sort_values(["ticker", "date"])
    daily.to_csv(DAILY_PRICES_FILE, index=False, compression="gzip")
    pd.DataFrame(failures).to_csv(DATA_DIR / "download_failures.csv", index=False)
    print(f"Saved {DAILY_PRICES_FILE} with {len(daily):,} rows and {daily['ticker'].nunique():,} tickers")
    if failures:
        print(f"Download failures/missing tickers: {len(failures):,}. See data/download_failures.csv")


if __name__ == "__main__":
    main()
