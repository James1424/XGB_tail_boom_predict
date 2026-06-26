import argparse
import io
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import DATA_DIR, SOURCE_ETFS_FILE, MANUAL_TICKERS_FILE, SEED_UNIVERSE_FILE, BENCHMARK

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_NDX = "https://en.wikipedia.org/wiki/Nasdaq-100"
ARK_URL_TEMPLATE = "https://ark-funds.com/wp-content/fundsiteliterature/csv/ARK_{name}_ETF_{ticker}_HOLDINGS.csv"
ARK_NAME = {"ARKK": "INNOVATION", "ARKW": "NEXT_GENERATION_INTERNET", "ARKG": "GENOMIC_REVOLUTION", "ARKQ": "AUTONOMOUS_TECHNOLOGY_ROBOTICS"}

# Known public CSV endpoints. They can change; failed URLs are skipped and reported.
HOLDINGS_URLS = {
    "QQQ": ["https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?audienceType=Investor&action=download&ticker=QQQ"],
    "SOXQ": ["https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?audienceType=Investor&action=download&ticker=SOXQ"],
    "TAN": ["https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?audienceType=Investor&action=download&ticker=TAN"],
    "IPO": ["https://www.renaissancecapital.com/Content/Images/IPOETF/indexholdings/IPO.csv"],
    # iShares AJAX CSV endpoints. Product URLs occasionally change, so failures are tolerated.
    "IWF": ["https://www.ishares.com/us/products/239706/ishares-russell-1000-growth-etf/1467271812596.ajax?fileType=csv&fileName=IWF_holdings&dataType=fund"],
    "IWO": ["https://www.ishares.com/us/products/239710/ishares-russell-2000-growth-etf/1467271812596.ajax?fileType=csv&fileName=IWO_holdings&dataType=fund"],
    "SOXX": ["https://www.ishares.com/us/products/239705/ishares-semiconductor-etf/1467271812596.ajax?fileType=csv&fileName=SOXX_holdings&dataType=fund"],
    "IGV": ["https://www.ishares.com/us/products/239771/ishares-expanded-tech-software-sector-etf/1467271812596.ajax?fileType=csv&fileName=IGV_holdings&dataType=fund"],
    "IBB": ["https://www.ishares.com/us/products/239699/ishares-biotechnology-etf/1467271812596.ajax?fileType=csv&fileName=IBB_holdings&dataType=fund"],
    "ICLN": ["https://www.ishares.com/us/products/239738/ishares-global-clean-energy-etf/1467271812596.ajax?fileType=csv&fileName=ICLN_holdings&dataType=fund"],
}


def normalize_ticker(x: str) -> str | None:
    if x is None:
        return None
    s = str(x).strip().upper()
    if not s or s in {"NAN", "-", "--", "CASH", "USD"}:
        return None
    s = s.replace("/", "-").replace(".", "-")
    if not re.match(r"^[A-Z][A-Z0-9-]{0,8}$", s):
        return None
    # Exclude non-common security suffixes and obvious placeholders.
    bad_fragments = ["-WS", "-WT", "-W", "-U", "-RT", "-R", "-P", "PRN", "CASH"]
    if any(s.endswith(b) for b in bad_fragments):
        return None
    return s


def extract_tickers_from_table(df: pd.DataFrame) -> list[str]:
    candidates = []
    preferred = ["ticker", "symbol", "holding ticker", "ticker symbol", "identifier"]
    for col in df.columns:
        c = str(col).strip().lower()
        if c in preferred or "ticker" in c or c == "symbol":
            candidates.extend(df[col].dropna().astype(str).tolist())
    # Fallback: scan all object columns, but keep strict ticker pattern.
    if not candidates:
        for col in df.columns:
            if df[col].dtype == object:
                candidates.extend(df[col].dropna().astype(str).tolist())
    out = []
    for x in candidates:
        t = normalize_ticker(x)
        if t:
            out.append(t)
    return sorted(set(out))


def fetch_csv_tickers(url: str) -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=45)
    r.raise_for_status()
    text = r.text
    # Some providers add metadata lines before the actual CSV header. Try likely header rows.
    lines = text.splitlines()
    for i in range(min(30, len(lines))):
        sample = "\n".join(lines[i:])
        try:
            df = pd.read_csv(io.StringIO(sample))
            ticks = extract_tickers_from_table(df)
            if len(ticks) >= 3:
                return ticks
        except Exception:
            pass
    return []


def _read_html_tables_with_headers(url: str) -> list[pd.DataFrame]:
    """Read HTML tables through requests with browser-like headers.

    GitHub Actions and other cloud runners are often blocked by sites such as
    Wikipedia when pandas opens the URL directly without a User-Agent.
    Downloading the HTML first with requests makes the holdings builder much
    more robust and keeps source failures non-fatal.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ETFPanelBuilder/1.0; +https://github.com/James1424/panel_label_construct)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get(url, headers=headers, timeout=45)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))


def fetch_wikipedia_sp500() -> list[str]:
    tables = _read_html_tables_with_headers(WIKI_SP500)
    for df in tables:
        if "Symbol" in df.columns:
            return [t for t in (normalize_ticker(x) for x in df["Symbol"]) if t]
    return []


def fetch_wikipedia_ndx() -> list[str]:
    tables = _read_html_tables_with_headers(WIKI_NDX)
    best = []
    for df in tables:
        ticks = extract_tickers_from_table(df)
        if len(ticks) > len(best):
            best = ticks
    return best


def get_source_tickers(etf: str, provider_hint: str) -> tuple[list[str], str]:
    etf = etf.upper()
    if etf == "SPY" or provider_hint == "wikipedia_sp500":
        try:
            ticks = fetch_wikipedia_sp500()
            return ticks, "wikipedia_sp500"
        except Exception as exc:
            print(f"Warning: Wikipedia S&P 500 fetch failed for {etf}: {exc}")
            return [], "failed_wikipedia_sp500"
    if etf == "QQQ":
        # Prefer official Invesco, fallback Wikipedia NDX.
        for url in HOLDINGS_URLS.get(etf, []):
            try:
                ticks = fetch_csv_tickers(url)
                if len(ticks) >= 50:
                    return ticks, "invesco_csv"
            except Exception:
                pass
        try:
            return fetch_wikipedia_ndx(), "wikipedia_nasdaq100_fallback"
        except Exception as exc:
            print(f"Warning: Wikipedia Nasdaq-100 fallback failed for {etf}: {exc}")
            return [], "failed_wikipedia_nasdaq100"
    if etf.startswith("ARK") and etf in ARK_NAME:
        url = ARK_URL_TEMPLATE.format(name=ARK_NAME[etf], ticker=etf)
        try:
            return fetch_csv_tickers(url), "ark_csv"
        except Exception:
            return [], "failed_ark_csv"
    for url in HOLDINGS_URLS.get(etf, []):
        try:
            ticks = fetch_csv_tickers(url)
            if ticks:
                return ticks, "csv_url"
        except Exception:
            continue
    return [], "unavailable"


def build_seed_universe() -> pd.DataFrame:
    if not SOURCE_ETFS_FILE.exists():
        raise FileNotFoundError(f"Missing {SOURCE_ETFS_FILE}")
    sources = pd.read_csv(SOURCE_ETFS_FILE)
    records = []
    failures = []
    for _, r in sources.iterrows():
        etf = str(r["ticker"]).upper()
        category = str(r["category"])
        weight = float(r.get("source_weight", 1))
        provider_hint = str(r.get("provider_hint", ""))
        ticks, method = get_source_tickers(etf, provider_hint)
        if not ticks:
            failures.append({"source": etf, "category": category, "method": method})
        for t in ticks:
            records.append({"ticker": t, "source": etf, "category": category, "source_weight": weight, "method": method})

    if MANUAL_TICKERS_FILE.exists():
        manual = pd.read_csv(MANUAL_TICKERS_FILE)
        for _, r in manual.iterrows():
            t = normalize_ticker(r.get("ticker"))
            if t:
                records.append({
                    "ticker": t,
                    "source": str(r.get("source", "manual")),
                    "category": str(r.get("category", "manual_core")),
                    "source_weight": float(r.get("source_weight", 1)),
                    "method": "manual_csv",
                })

    if not records:
        raise RuntimeError("No seed tickers found. Check ETF holding URLs or add data/manual_tickers.csv.")

    long = pd.DataFrame(records).drop_duplicates(["ticker", "source", "category"])
    categories = sorted(long["category"].dropna().unique().tolist())
    summary_rows = []
    for t, g in long.groupby("ticker"):
        cats = sorted(set(g["category"].astype(str)))
        row = {
            "ticker": t,
            "source_count": g["source"].nunique(),
            "source_weight_sum": g.drop_duplicates("source")["source_weight"].sum(),
            "theme_count": len(cats),
            "sources": ",".join(sorted(set(g["source"].astype(str)))),
            "categories": ",".join(cats),
        }
        for c in categories:
            row[f"in_{c}"] = int(c in cats)
        summary_rows.append(row)
    universe = pd.DataFrame(summary_rows).sort_values(["source_weight_sum", "source_count", "ticker"], ascending=[False, False, True])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    universe.to_csv(SEED_UNIVERSE_FILE, index=False)
    long.to_csv(DATA_DIR / "seed_universe_sources_long.csv", index=False)
    pd.DataFrame(failures).to_csv(DATA_DIR / "holding_source_failures.csv", index=False)
    print(f"Saved {SEED_UNIVERSE_FILE}: {len(universe):,} tickers from {long['source'].nunique():,} sources")
    if failures:
        print(f"Holding sources with no data: {len(failures)}. See data/holding_source_failures.csv")
    return universe


def main() -> None:
    build_seed_universe()


if __name__ == "__main__":
    main()
