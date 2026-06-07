"""
sp500_ceiling.py
================

Single-file research pipeline for measuring how point-in-time S&P 500
membership and missing delisted prices affect a monthly factor backtest.

The script has three jobs:
  1. Reconstruct historical S&P 500 membership from Wikipedia and fja05680.
  2. Build a daily price file and recover missing delisted histories with Tiingo.
  3. Build a monthly factor panel, run the backtest, and write a ticker registry
     that documents which historical names are covered, recovered, or absent.

Legacy web-scraping recovery sources are intentionally not used. Stooq, Nasdaq,
BigCharts, and Alpha Vantage created long failure runs without improving the
dataset. Tiingo is the only active recovery source because it provides an
authenticated historical end-of-day endpoint and can return delisted histories.

Usage:
  python sp500_ceiling.py
  python sp500_ceiling.py --no-download          use existing daily CSV
  python sp500_ceiling.py --no-recovery          skip Tiingo recovery
  python sp500_ceiling.py --tiingo-plan          show recovery queue; make no API calls
  python sp500_ceiling.py --tiingo-max-calls 25  cap Tiingo calls for this run
  python sp500_ceiling.py --edgar                opt into EDGAR terminal search
  python sp500_ceiling.py --bias-study           compare survivorship-bias layers
  python sp500_ceiling.py --no-ff                skip Fama-French download
  python sp500_ceiling.py --plots                save equity-curve PNGs
  python sp500_ceiling.py --dry-run              no writes and no Tiingo calls
  python sp500_ceiling.py --resume               skip tickers already in the CSV

Environment:
  TIINGO_API_KEY       Tiingo API token. TIINGO_TOKEN is also accepted.
  TIINGO_DAILY_LIMIT   Default: 1000. Used by the local quota guard.
  TIINGO_RESERVE       Default: 25. Calls to leave unused at the end of a run.
  SP500_PIPELINE_DATA_DIR
                       Optional folder for CSV inputs/outputs. Defaults to the
                       folder containing this script.
"""
from __future__ import annotations

import os, re, sys, time, random, warnings, zipfile, bisect, json
from collections import Counter
from datetime import date
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

try:
    import yfinance as yf;  _HAS_YF = True
except ImportError:
    _HAS_YF = False;  print("WARNING: yfinance not installed — download disabled.")

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt; _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from scipy.stats import norm as _norm; _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — CONFIGURATION & CLI
# ══════════════════════════════════════════════════════════════════════════════

_HERE          = Path(__file__).resolve().parent
DATA_DIR       = Path(os.environ.get('SP500_PIPELINE_DATA_DIR', _HERE)).resolve()
START_DATE     = '2000-01-01'
END_DATE       = '2030-12-31'
DAILY_FILE     = str(DATA_DIR / 'sp500_daily_clean.csv')
DAILY_TEMP     = DAILY_FILE + '.tmp'
FACTOR_FILE    = str(DATA_DIR / 'factors_clean.csv')
ANCHORS_FILE   = str(DATA_DIR / 'sp500_wiki_quarterly_anchors.csv')
FJA_FILE       = str(DATA_DIR / 'sp500_fja05680.csv')
REGISTRY_FILE  = str(DATA_DIR / 'missing_ticker_registry.csv')
TERMINAL_FILE  = str(DATA_DIR / 'terminal_prices.csv')
SI_CACHE_FILE  = str(DATA_DIR / 'short_interest_proxy.csv')
TIINGO_USAGE_FILE = str(DATA_DIR / 'tiingo_usage_ledger.json')

def _cli_value(flag: str, default: str | None = None) -> str | None:
    """Read --flag value or --flag=value from sys.argv."""
    prefix = flag + '='
    for i, arg in enumerate(sys.argv):
        if arg == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


def _cli_int(flag: str, default: int | None = None) -> int | None:
    raw = _cli_value(flag)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        print(f"  Ignoring invalid {flag} value: {raw!r}")
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


TIINGO_API_KEY     = os.environ.get('TIINGO_API_KEY') or os.environ.get('TIINGO_TOKEN', '')
TIINGO_DAILY_LIMIT = max(0, _env_int('TIINGO_DAILY_LIMIT', 1000))
TIINGO_RESERVE     = max(0, _env_int('TIINGO_RESERVE', 25))
TIINGO_DELAY       = max(0.0, _env_float('TIINGO_DELAY', 0.25))

IS_END      = pd.Timestamp('2014-12-31')
OOS_START   = pd.Timestamp('2015-01-01')
LONG_COST_BPS_OW = 20
COST_TIERS = {
    'A': {'label': 'Tier A (Institutional)', 'rt_bps': 10},
    'B': {'label': 'Tier B (Hedge Fund)',    'rt_bps': 30},
    'C': {'label': 'Tier C (Conservative)',  'rt_bps': 50},
}

FF5_URL = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
           'F-F_Research_Data_5_Factors_2x3_CSV.zip')

NO_DOWNLOAD  = '--no-download'  in sys.argv
NO_RECOVERY  = '--no-recovery'  in sys.argv
RUN_EDGAR    = '--edgar'        in sys.argv and '--no-edgar' not in sys.argv
RUN_OPENFIGI = '--openfigi'     in sys.argv
BIAS_STUDY   = '--bias-study'   in sys.argv
NO_FF        = '--no-ff'        in sys.argv
PLOTS        = '--plots'        in sys.argv
DRY_RUN      = '--dry-run'      in sys.argv
RESUME       = '--resume'       in sys.argv
SAVE_CSV     = '--csv'          in sys.argv
TIINGO_PLAN  = '--tiingo-plan' in sys.argv or '--recovery-plan' in sys.argv
TIINGO_MAX_CALLS = _cli_int('--tiingo-max-calls')
HEADERS = {
    'User-Agent': ('Mozilla/5.0 (compatible; SurvivorshipBiasResearch/1.0; '
                   '+mailto:your@email.com)'),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

# Known acquisition/terminal prices (hand-verified from SEC filings + public records)
# Format: ticker → (price_per_share, last_price_date, event_type, source)
CURATED_TERMINAL: dict[str, tuple[float, str, str, str]] = {
    'CELG': (102.43, '2019-11-20', 'acquisition', 'BMS merger 8-K 2019-11-20'),
    'ATVI': (95.00,  '2023-10-13', 'acquisition', 'MSFT merger 8-K 2023-10-13'),
    'RHT':  (190.00, '2019-07-09', 'acquisition', 'IBM merger 8-K 2019-07-09'),
    'MON':  (128.00, '2018-06-07', 'acquisition', 'Bayer 8-K 2018-06-07'),
    'WFM':  (42.00,  '2017-08-28', 'acquisition', 'Amazon 8-K 2017-08-28'),
    'ALTR': (54.00,  '2015-12-28', 'acquisition', 'Intel 8-K 2015-12-28'),
    'BRCM': (37.00,  '2016-02-01', 'acquisition', 'Avago 8-K 2016-02-01'),
    'CTXS': (104.00, '2022-09-30', 'acquisition', 'Vista/Elliott 8-K 2022-09-30'),
    'MXIM': (85.61,  '2021-08-26', 'acquisition', 'ADI 8-K 2021-08-26'),
    'XLNX': (143.00, '2022-02-14', 'acquisition', 'AMD 8-K 2022-02-14'),
    'LLTC': (60.00,  '2017-03-10', 'acquisition', 'ADI 8-K 2017-03-10'),
    'AGN':  (199.25, '2020-05-08', 'acquisition', 'AbbVie 8-K 2020-05-08'),
    'LLL':  (18.23,  '2019-06-29', 'merger',      'L3Harris 8-K 2019-06-29 (share-for-share)'),
    'RTN':  (None,   '2020-04-03', 'merger',      'Share-for-share RTX merger; no cash consideration'),
    'LVLT': (26.50,  '2017-11-01', 'acquisition', 'CenturyLink 8-K 2017-11-01'),
    'JOY':  (28.30,  '2017-07-03', 'acquisition', 'Komatsu 8-K 2017-07-03'),
    'PCP':  (235.00, '2016-01-29', 'acquisition', 'Berkshire 8-K 2016-01-29'),
    'VAR':  (177.25, '2021-04-15', 'acquisition', 'Siemens 8-K 2021-04-15'),
    'ABMD': (380.00, '2023-01-01', 'acquisition', 'J&J 8-K 2022-12-23'),
    'COV':  (102.00, '2015-01-26', 'acquisition', 'Medtronic 8-K 2015-01-26'),
    'BCR':  (222.93, '2017-10-27', 'acquisition', 'BD 8-K 2017-10-27'),
    'DTV':  (95.00,  '2015-07-24', 'acquisition', 'AT&T 8-K 2015-07-24'),
    'STJ':  (85.00,  '2017-01-04', 'acquisition', 'Abbott 8-K 2017-01-04'),
    'SNI':  (87.25,  '2018-03-31', 'acquisition', 'Discovery 8-K 2018-03-31'),
    'RAI':  (58.12,  '2017-07-25', 'acquisition', 'BAT 8-K 2017-07-25'),
    'SIAL': (150.00, '2015-11-18', 'acquisition', 'Merck KGaA 8-K 2015-11-18'),
    'KFT':  (None,   '2012-10-01', 'spin-off',    'Split into Mondelez (MDLZ) + Kraft; no cash'),
    'DWDP': (None,   '2019-04-01', 'spin-off',    'Three-way split DD/DOW/CTVA; no cash'),
    'HNZ':  (72.50,  '2013-06-07', 'acquisition', 'Berkshire/3G 8-K 2013-06-07'),
    'GMCR': (92.00,  '2016-03-17', 'acquisition', 'JAB 8-K 2016-03-17'),
    'GGP':  (23.50,  '2018-08-28', 'acquisition', 'Brookfield 8-K 2018-08-28'),
    'TSS':  (119.86, '2019-09-18', 'acquisition', 'FIS 8-K 2019-09-18'),
    'NLSN': (28.00,  '2022-10-11', 'acquisition', 'private equity 8-K 2022-10-11'),
    'DISCA': (43.00, '2022-04-08', 'merger',      'WBD merger; $43 implied value'),
    'DISCK': (43.00, '2022-04-08', 'merger',      'WBD merger; $43 implied value'),
    'TWC':  (195.71, '2016-05-18', 'acquisition', 'Charter 8-K 2016-05-18'),
    # Bankruptcies — terminal price ≈ last traded price before halt/delisting
    'LEH':  (0.21,   '2008-09-15', 'bankruptcy',  'NYSE halt; last reported trade ~$0.21'),
    'EK':   (0.36,   '2012-01-19', 'bankruptcy',  'NYSE delist; last trade ~$0.36'),
    'ABK':  (0.48,   '2010-11-08', 'bankruptcy',  'NYSE delist; last trade ~$0.48'),
    'FNM':  (0.44,   '2010-06-16', 'conservatorship', 'NYSE delist; last trade ~$0.44'),
    'FRE':  (0.34,   '2010-06-16', 'conservatorship', 'NYSE delist; last trade ~$0.34'),
    'CHK':  (0.01,   '2020-06-26', 'bankruptcy',  'NYSE delist; ~$0.01 last trade'),
    'WIN':  (0.06,   '2019-02-25', 'bankruptcy',  'OTC; last trade ~$0.06'),
    'FTR':  (0.98,   '2020-04-14', 'bankruptcy',  'Nasdaq delist; ~$0.98'),
    'ENDP': (0.36,   '2022-08-17', 'bankruptcy',  'Nasdaq delist; ~$0.36'),
    'DNR':  (0.27,   '2020-07-30', 'bankruptcy',  'NYSE delist; ~$0.27'),
}

# Same-entity ticker changes only. Acquisitions and mergers are excluded because
# the acquirer's post-deal prices are not a valid continuation of the target's
# history. Those cases belong in terminal price injection instead.
KNOWN_RENAMES: dict[str, str] = {
    'JDSU': 'VIAV',   # Viavi Solutions — same legal entity, ticker renamed 2015
    'PCLN': 'BKNG',   # Booking Holdings — same legal entity, ticker renamed 2018
}

REQUEST_DELAY = 1.2
MAX_RETRIES   = 4

SIGNAL_SPECS = [
    ('Base  mom−vol+vchg  ★',  1.0, -1.0,  1.0),
    ('Momentum only',          1.0,  0.0,  0.0),
    ('Vol-change only',        0.0,  0.0,  1.0),
    ('mom + volchg (no vol)',  1.0,  0.0,  1.0),
    ('2×mom − vol + vchg',    2.0, -1.0,  1.0),
]
DECILE_COUNTS = [5, 8, 10, 20]
W = 72


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _safe_get(url: str, timeout: int = 20, headers: dict | None = None,
              retries: int = MAX_RETRIES) -> requests.Response | None:
    """HTTP GET with exponential back-off on 429/transient errors."""
    hdrs = headers if headers is not None else HEADERS
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=hdrs, timeout=timeout)
            if r.status_code == 429:
                wait = 2 ** attempt + random.uniform(0, 1)
                time.sleep(wait)
                continue
            return r
        except requests.exceptions.RequestException:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def _standardise_ohlcv(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalise a yfinance DataFrame to 7-column schema within [START_DATE, END_DATE]."""
    df = raw.reset_index().rename(columns={'index': 'Date', 'Datetime': 'Date'})
    df['Ticker'] = ticker
    for col in ('Open', 'High', 'Low', 'Close', 'Volume'):
        if col not in df.columns:
            df[col] = np.nan
    df = df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    dates = pd.to_datetime(df['Date'])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert(None)
    df['Date'] = dates
    df = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)]
    df.dropna(subset=['Close'], inplace=True)
    return df.reset_index(drop=True)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fuzzy-rename columns to {Date, Open, High, Low, Close, Volume}."""
    rename = {}
    for col in df.columns:
        cl = str(col).strip().lower()
        if 'date' in cl:                                              rename[col] = 'Date'
        elif 'open' in cl:                                            rename[col] = 'Open'
        elif 'high' in cl:                                            rename[col] = 'High'
        elif 'low' in cl and 'close' not in cl and 'flow' not in cl: rename[col] = 'Low'
        elif 'close' in cl or 'last' in cl or cl == 'adj close':     rename[col] = 'Close'
        elif 'vol' in cl:                                             rename[col] = 'Volume'
    return df.rename(columns=rename)


def _standardise_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Full normalise → date-filter → drop-no-close pipeline for scraped data."""
    df = _normalise_columns(df.copy())
    df['Ticker'] = ticker
    df['Date'] = pd.to_datetime(df.get('Date', pd.NaT), errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['Date'])
    for col in ('Open', 'High', 'Low', 'Close', 'Volume'):
        if col not in df.columns:
            df[col] = np.nan
        else:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)]
    return df.dropna(subset=['Close']).sort_values('Date').reset_index(drop=True)

def _extract_tickers_from_soup(soup: BeautifulSoup) -> set[str] | None:
    """
    Parse S&P 500 ticker symbols from any Wikipedia-formatted page.
    Older revisions use inconsistent table layouts, so this tries progressively
    looser table-detection rules before giving up.
    """
    # Strategy 1: id='constituents'
    table = soup.find('table', {'id': 'constituents'})
    if table is None:
        # Strategy 2: wikitable class with symbol/ticker/tick header
        for t in soup.find_all('table', class_=re.compile(r'wikitable', re.I)):
            ths = [th.get_text(strip=True).lower() for th in t.find_all('th')[:6]]
            if any(h in ('symbol', 'ticker', 'tick') for h in ths):
                table = t
                break
    if table is None:
        # Strategy 3: any table with a caption containing 'S&P' or 'constituent'
        for t in soup.find_all('table'):
            cap = t.find('caption')
            if cap and ('s&p' in cap.get_text(strip=True).lower()
                        or 'constituent' in cap.get_text(strip=True).lower()):
                table = t
                break
    if table is None:
        # Strategy 4: fallback to the largest table on the page (by row count)
        tables = soup.find_all('table')
        if tables:
            best = max(tables, key=lambda t: len(t.find_all('tr')))
            if len(best.find_all('tr')) > 30:
                table = best

    if table is None:
        return None

    try:
        df = pd.read_html(StringIO(str(table)))[0]
    except Exception:
        return None

    # Try to find a column that looks like tickers (uppercase letters + dash, length 1–6)
    def is_ticker_col(col_values) -> bool:
        clean = col_values.dropna().astype(str).str.strip()
        if len(clean) < 10:
            return False
        matches = clean.str.match(r'^[A-Z\.\-]{1,6}$')
        return matches.mean() > 0.7

    ticker_candidates = [c for c in df.columns if is_ticker_col(df[c])]
    if ticker_candidates:
        sym_col = ticker_candidates[0]
    else:
        # Fallback to header keyword matching
        sym_col = next((c for c in df.columns
                        if any(kw in str(c).lower() for kw in ('symbol', 'ticker', 'tick'))),
                       df.columns[0])

    tickers = set()
    for raw in df[sym_col].dropna():
        t = str(raw).strip().upper().replace('.', '-').split()[0]
        if t and re.match(r'^[A-Z\-]{1,6}$', t):
            tickers.add(t)
    return tickers if len(tickers) > 10 else None



def _is_html(text: str) -> bool:
    head = text.lstrip()[:120].lower()
    return head.startswith('<!') or '<html' in head or '<head' in head


def _find_csv_header(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        lo = line.lower().replace(' ', '')
        if 'date' in lo and any(k in lo for k in ('open', 'high', 'low', 'close', 'price')):
            return i
    return None


def _perf(rets: pd.Series, label: str = '') -> dict:
    """Compute standard performance metrics for a return series."""
    rets = rets.dropna()
    if len(rets) < 2:
        return {}
    cum      = (1 + rets).cumprod()
    years    = len(rets) / 12
    cagr     = cum.iloc[-1] ** (1.0 / years) - 1 if years > 0 else np.nan
    sharpe   = np.sqrt(12) * rets.mean() / rets.std() if rets.std() else 0.0
    max_dd   = (cum / cum.cummax() - 1).min()
    hit      = (rets > 0).mean()
    avg_win  = rets[rets > 0].mean() if (rets > 0).any() else 0.0
    avg_loss = rets[rets < 0].mean() if (rets < 0).any() else 0.0
    return dict(label=label, n=len(rets), cagr=cagr, sharpe=sharpe,
                max_dd=max_dd, hit=hit, avg_win=avg_win, avg_loss=avg_loss)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MEMBERSHIP DATABASE  (fja05680 + Wikipedia revision API)
# ══════════════════════════════════════════════════════════════════════════════

FJA_GITHUB_API = ('https://api.github.com/repos/fja05680/sp500/'
                  'git/trees/master?recursive=1')
FJA_BASE       = 'https://raw.githubusercontent.com/fja05680/sp500/master/'
WIKI_API       = 'https://en.wikipedia.org/w/api.php'

def load_fja05680() -> dict[pd.Timestamp, set[str]]:
    """Download fja05680 historical S&P 500 constituent CSV from GitHub."""
    print("\n[Membership] Loading fja05680 dataset…")
    csv_url = None

    api_r = _safe_get(FJA_GITHUB_API,
                      headers={**HEADERS, 'Accept': 'application/vnd.github.v3+json'})
    if api_r and api_r.status_code == 200:
        try:
            for node in api_r.json().get('tree', []):
                p = node.get('path', '')
                if 'S&P 500 Historical' in p and p.endswith('.csv'):
                    csv_url = FJA_BASE + requests.utils.quote(p, safe='/()')
                    break
        except Exception:
            pass

    if csv_url is None:
        for suffix in ['(07-22-2023)', '(08-01-2024)', '(2024)', '(2025)', '(2026)']:
            trial = FJA_BASE + requests.utils.quote(
                f"S&P 500 Historical Components & Changes{suffix}.csv", safe='()')
            r = _safe_get(trial)
            if r and r.status_code == 200 and len(r.content) > 5_000:
                csv_url = trial; break

    if csv_url is None:
        print("  ✗ fja05680: could not locate CSV.")
        return {}

    r = _safe_get(csv_url)
    if r is None or r.status_code != 200:
        return {}
    print(f"  ✓ fja05680: {csv_url.split('/')[-1][:60]}")

    try:
        df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=[0])
    except Exception as e:
        print(f"  ✗ Parse error: {e}"); return {}

    membership: dict[pd.Timestamp, set[str]] = {}
    ticker_cols = [c for c in df.columns if re.match(r'^[A-Z\.\-]{1,6}$', str(c))]
    if len(ticker_cols) > 50:
        for date_idx, row in df.iterrows():
            ts = pd.Timestamp(date_idx) + pd.offsets.MonthEnd(0)
            members = {c.replace('.', '-').upper() for c in ticker_cols if row.get(c, 0) == 1}
            if members:
                membership[ts] = members
    else:
        for date_idx, row in df.iterrows():
            ts = pd.Timestamp(date_idx) + pd.offsets.MonthEnd(0)
            raw = str(row.iloc[0]) if len(row) > 0 else ''
            tickers = {t.strip().upper().replace('.', '-')
                       for t in re.split(r'[,\s]+', raw)
                       if re.match(r'^[A-Z\-]{1,5}$', t.strip().upper())}
            if tickers:
                membership[ts] = tickers

    print(f"  ✓ fja05680: {len(membership)} monthly snapshots")
    return membership


# ── Wikipedia revision-history anchors ────────────────────────────────────────

def _fetch_wiki_revision_ids() -> list[tuple[pd.Timestamp, int]]:
    """
    Paginate through ALL revisions of 'List_of_S&P_500_companies' since 2000.
    Returns [(timestamp, revid), …] sorted by timestamp ascending.

    The title is passed as raw text so requests can encode the ampersand once.
    Revision timestamps are converted to timezone-naive pandas timestamps to
    match the quarterly anchor dates used by bisect.
    """
    revisions: list[tuple[pd.Timestamp, int]] = []
    params = {
        'action':   'query',
        'titles':   'List_of_S&P_500_companies',
        'prop':     'revisions',
        'rvprop':   'ids|timestamp',
        'rvlimit':  'max',
        'rvdir':    'newer',
        'rvstart':  '2000-01-01T00:00:00Z',
        'format':   'json',
    }
    while True:
        try:
            r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                break
            data = r.json()
            pages = data.get('query', {}).get('pages', {})
            page  = next(iter(pages.values()))
            for rev in page.get('revisions', []):
                ts    = pd.Timestamp(rev['timestamp']).tz_localize(None)
                revid = int(rev['revid'])
                revisions.append((ts, revid))
            cont = data.get('continue', {}).get('rvcontinue')
            if not cont:
                break
            params['rvcontinue'] = cont
            time.sleep(0.3)
        except Exception:
            break
    revisions.sort(key=lambda x: x[0])
    return revisions


def _fetch_wiki_at_revision(oldid: int) -> set[str] | None:
    """Fetch the Wikipedia S&P 500 article at a specific revision and extract tickers."""
    url = (f"https://en.wikipedia.org/w/index.php"
           f"?title=List_of_S%26P_500_companies&oldid={oldid}")
    r = _safe_get(url, timeout=30)
    if r is None or r.status_code != 200:
        return None
    soup = BeautifulSoup(r.text, 'html.parser')
    return _extract_tickers_from_soup(soup)


def build_quarterly_wiki_anchors(years: list[int]) -> dict[pd.Timestamp, set[str]]:
    """
    Fetch quarterly S&P 500 snapshots from Wikipedia's revision history.
    For each quarter-end target date, use the latest page revision available
    on or before that date.
    """
    anchors: dict[pd.Timestamp, set[str]] = {}

    targets: list[pd.Timestamp] = []
    for year in years:
        for m, d in [(3, 31), (6, 30), (9, 30), (12, 31)]:
            targets.append(pd.Timestamp(f'{year}-{m:02d}-{d:02d}'))
    targets = [t for t in targets if t >= pd.Timestamp('2001-01-01')]

    print(f"\n[Membership] Wikipedia revision API — "
          f"{len(targets)} quarterly targets ({years[0]}–{years[-1]})…")

    print("  Fetching revision history (paginated)…", end=' ', flush=True)
    rev_list = _fetch_wiki_revision_ids()
    if not rev_list:
        print("✗ failed"); return {}
    rev_timestamps = [r[0] for r in rev_list]
    rev_ids        = [r[1] for r in rev_list]
    print(f"✓ {len(rev_list):,} revisions since 2000")

    fetched_revids: dict[int, set[str] | None] = {}

    for target in targets:
        idx = bisect.bisect_right(rev_timestamps, target) - 1
        if idx < 0:
            continue
        revid = rev_ids[idx]

        if revid not in fetched_revids:
            tickers = _fetch_wiki_at_revision(revid)
            fetched_revids[revid] = tickers
            time.sleep(0.5)

        tickers = fetched_revids[revid]
        if tickers and len(tickers) > 400:
            anchors[target] = tickers
            ts_of_rev = rev_timestamps[idx]
            lag_days  = (target - ts_of_rev).days
            print(f"  {target.date()}  ✓  {len(tickers)} tickers  "
                  f"(revision {revid}, {lag_days}d before target)")
        elif tickers:
            print(f"  {target.date()}  ⚠  only {len(tickers)} tickers — skipped")
        else:
            print(f"  {target.date()}  ✗  parse failed")

    print(f"  → {len(anchors)}/{len(targets)} quarterly anchors recovered")
    return anchors


def build_membership_from_wikipedia() -> tuple[dict[pd.Timestamp, set[str]], set[str], pd.DataFrame]:
    """
    Reconstruct S&P 500 membership by walking backwards from today's composition.
    Returns (membership_sets, current_tickers, history_df).
    """
    print("\n[Membership] Scraping Wikipedia S&P 500 article…")
    resp = requests.get(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        headers=HEADERS, timeout=30)
    soup = BeautifulSoup(resp.text, 'html.parser')

    cur_table = soup.find('table', {'id': 'constituents'})
    cur_df    = pd.read_html(StringIO(str(cur_table)))[0]
    current_tickers = set(cur_df['Symbol'].str.replace('.', '-').str.upper())

    for h2 in soup.find_all('h2'):
        if 'Selected changes' in h2.get_text():
            changes_table = h2.find_next('table', class_='wikitable')
            break
    changes_df = pd.read_html(StringIO(str(changes_table)))[0]
    changes_df.columns = [
        ' '.join([str(c) for c in col if 'Unnamed' not in str(c)]).strip()
        for col in changes_df.columns]
    date_col    = [c for c in changes_df.columns if 'Date'    in c][0]
    added_col   = [c for c in changes_df.columns if 'Added'   in c and 'Ticker' in c][0]
    removed_col = [c for c in changes_df.columns if 'Removed' in c and 'Ticker' in c][0]
    changes_df[date_col] = pd.to_datetime(changes_df[date_col], errors='coerce')
    changes_df = changes_df.dropna(subset=[date_col])

    records = []
    for _, row in changes_df.iterrows():
        d = row[date_col]
        if pd.notna(row[added_col]):
            for t in str(row[added_col]).replace('\n', ' ').split():
                t = t.strip().upper().replace('.', '-')
                if t and t != '—':
                    records.append({'date': d, 'ticker': t, 'action': 'added'})
        if pd.notna(row[removed_col]):
            for t in str(row[removed_col]).replace('\n', ' ').split():
                t = t.strip().upper().replace('.', '-')
                if t and t != '—':
                    records.append({'date': d, 'ticker': t, 'action': 'removed'})
    history = pd.DataFrame(records).sort_values('date')

    all_dates = [pd.Timestamp.today()] + sorted(history['date'].unique(), reverse=True)
    membership_sets: dict[pd.Timestamp, set[str]] = {}
    current_set = current_tickers.copy()
    for d in all_dates:
        for _, ch in history[history['date'] == d].iterrows():
            if ch['action'] == 'added':
                current_set.discard(ch['ticker'])
            else:
                current_set.add(ch['ticker'])
        membership_sets[d] = current_set.copy()

    print(f"  ✓ Reconstructed {len(membership_sets)} membership snapshots")
    return membership_sets, current_tickers, history


def get_members_factory(membership_sets: dict[pd.Timestamp, set[str]],
                        anchors: dict[pd.Timestamp, set[str]],
                        tolerance_days: int = 5):
    """
    Build a point-in-time get_members() function that:
    1. For dates within tolerance_days of a quarterly anchor, returns the anchor set
    2. Otherwise falls back to the backward-reconstruction set
    """
    sorted_membership = sorted(membership_sets.keys())
    sorted_anchors    = sorted(anchors.items()) if anchors else []

    def get_members(as_of_date: pd.Timestamp) -> set[str]:
        ts = pd.Timestamp(as_of_date)
        for anchor_ts, snap_set in sorted_anchors:
            if abs((ts - anchor_ts).days) <= tolerance_days:
                return snap_set
        idx = bisect.bisect_left(sorted_membership, ts)
        if idx < len(sorted_membership):
            return membership_sets[sorted_membership[idx]]
        return set()

    return get_members


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MISSING-TICKER DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

MISSING_TICKERS_HARDCODED: list[str] = [
    'ABK','ABMD','ACAS','ACE','ADS','AGN','AKS','ALTR','ALXN','ANR',
    'ANSS','APOL','ARG','ARNC','ATVI','AV','AVP','AYE','BCR','BHI',
    'BIG','BJS','BRCM','BS','BXLT','CELG','CEPH','CERN','CFN','CHK',
    'CMA','CMCSK','COV','CPGX','CSC','CTLT','CTX','CTXS','CVC','CVH',
    'CXO','DAY','DF','DFS','DISCA','DISCK','DISH','DJ','DNB','DNR',
    'DO','DPS','DRE','DTV','DWDP','EK','ENDP','ESV','ETFC','FBHS',
    'FDC','FDO','FII','FL','FLIR','FNM','FRC','FRE','FRX','FTR',
    'GAS','GGP','GMCR','GPS','GRA','HBI','HCBK','HES','HFC','HNG',
    'HNZ','HPH','HSP','IGT','IPG','JCP','JDSU','JNPR','JNS','JNY',
    'JOY','JWN','K','KFT','KRFT','KSE','KSU','LDW','LEH','LLL',
    'LLTC','LM','LO','LSI','LVLT','LXK','MDP','MFE','MIL','MJN',
    'MNK','MON','MRO','MWW','MXIM','NBL','NCR','NLSN','NOVL','NVLS',
    'NYX','ODP','OMX','PBCT','PCP','PDCO','PETM','PGN','PLL','PXD',
    'QEP','QTRN','RAD','RAI','RDC','RHT','RRD','RTN','SAI','SBL',
    'SIAL','SIVB','SLR','SMS','SNI','SRCL','STJ','STR','SWN','SWY',
    'TGNA','TIF','TRB','TSG','TSS','TWC','TWTR','TYC','VAR','VIAB',
    'WBA','WCG','WFM','WFR','WIN','WPX','WYN','X','XEC','XL','XLNX',
    'XTO','YHOO',
]


def discover_unknown_unknowns(all_historical: set[str],
                               tickers_in_csv: set[str],
                               hardcoded_missing: list[str],
                               daily_df: pd.DataFrame,
                               min_rows: int = 120) -> list[str]:
    """
    Find historical S&P 500 tickers that are absent from the daily CSV or have
    too little price history to support the factor calculations.
    """
    known = set(hardcoded_missing)
    absent = all_historical - tickers_in_csv - known

    if not daily_df.empty:
        row_counts   = daily_df.groupby('Ticker').size()
        thin_tickers = set(row_counts[row_counts < min_rows].index)
        thin_unknown = (thin_tickers & all_historical) - known - absent
    else:
        thin_unknown = set()

    unknown_unknowns = sorted(absent | thin_unknown)

    if unknown_unknowns:
        print(f"\n[Discovery] Found {len(unknown_unknowns)} missing or thin-history tickers:")
        print(f"  Absent from CSV       : {len(absent)}")
        print(f"  Thin data (<{min_rows} rows) : {len(thin_unknown)}")
        print(f"  Tickers: {', '.join(sorted(unknown_unknowns)[:30])}"
              f"{'…' if len(unknown_unknowns) > 30 else ''}")
    else:
        print("\n[Discovery] No additional missing tickers found.")

    return unknown_unknowns


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PRICE DOWNLOAD  (Yahoo Finance)
# ══════════════════════════════════════════════════════════════════════════════

def _dl_ticker_backoff(ticker: str, max_retries: int = 5) -> pd.DataFrame:
    """Download full OHLCV history for a single ticker with exponential back-off."""
    for attempt in range(max_retries):
        try:
            data = yf.Ticker(ticker).history(period='max', auto_adjust=True)
            return data
        except Exception as exc:
            err = str(exc).lower()
            is_rl = '429' in err or 'rate limit' in err or 'too many' in err
            if is_rl and attempt < max_retries - 1:
                wait = 2 ** attempt + random.uniform(0.0, 1.0)
                time.sleep(wait)
            else:
                return pd.DataFrame()
    return pd.DataFrame()


def download_all_prices(active_tickers: set[str],
                        delisted_tickers: set[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Download Yahoo price history for all historical tickers.
    Returns (combined_df, missing_tickers_list).
    """
    if not _HAS_YF:
        return pd.DataFrame(), list(active_tickers | delisted_tickers)

    if os.path.exists(DAILY_TEMP):
        os.remove(DAILY_TEMP)
    pd.DataFrame(columns=['Date','Ticker','Open','High','Low','Close','Volume']
                 ).to_csv(DAILY_TEMP, index=False)

    def append_csv(df: pd.DataFrame) -> None:
        df.to_csv(DAILY_TEMP, mode='a', header=False, index=False)

    missing: list[str] = []

    print(f"\n[Download] Active tickers batch ({len(active_tickers)})…")
    data_active = yf.download(
        list(active_tickers), start=START_DATE, end=END_DATE,
        auto_adjust=True, group_by='ticker', threads=True, progress=True)
    for t in active_tickers:
        try:
            if t in data_active.columns.get_level_values(0).unique():
                df_t = _standardise_ohlcv(data_active[t].copy(), t)
                if not df_t.empty:
                    append_csv(df_t); continue
        except Exception:
            pass
        raw = _dl_ticker_backoff(t)
        if not raw.empty:
            df_t = _standardise_ohlcv(raw, t)
            if not df_t.empty:
                append_csv(df_t); continue
        missing.append(t)

    print(f"\n[Download] Delisted tickers individual ({len(delisted_tickers)})…")
    for i, t in enumerate(sorted(delisted_tickers)):
        raw = _dl_ticker_backoff(t)
        if not raw.empty:
            df_t = _standardise_ohlcv(raw, t)
            if not df_t.empty:
                append_csv(df_t)
            else:
                missing.append(t)
        else:
            missing.append(t)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(delisted_tickers)} delisted processed…")
        time.sleep(0.05)

    # Keep fresh Yahoo rows when they overlap old rows, while preserving
    # recovered-only rows from prior runs.
    new_df = pd.read_csv(DAILY_TEMP, parse_dates=['Date'])
    if os.path.exists(DAILY_FILE):
        existing_df = pd.read_csv(DAILY_FILE, parse_dates=['Date'])
        combined = (pd.concat([existing_df, new_df], ignore_index=True)
                    .drop_duplicates(subset=['Date', 'Ticker'], keep='last')
                    .sort_values(['Ticker', 'Date']).reset_index(drop=True))
        n_preserved = len(combined) - len(new_df)
        if n_preserved > 0:
            print(f"  Preserved {n_preserved:,} recovered rows from prior recovery run.")
    else:
        combined = new_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    tmp = DAILY_FILE + '.final_tmp'
    combined.to_csv(tmp, index=False)
    os.replace(tmp, DAILY_FILE)
    os.remove(DAILY_TEMP)
    print(f"  Daily CSV → {DAILY_FILE}  ({os.path.getsize(DAILY_FILE)/1e6:.1f} MB)")
    return combined, missing


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MISSING TICKER RECOVERY  (Tiingo)
# ══════════════════════════════════════════════════════════════════════════════

TIINGO_PRICE_URL = 'https://api.tiingo.com/tiingo/daily/{ticker}/prices'
EMPTY_PRICE_FRAME = pd.DataFrame(columns=['Date', 'Ticker', 'Open', 'High',
                                          'Low', 'Close', 'Volume'])


def _tiingo_today() -> str:
    return date.today().isoformat()


def _load_tiingo_usage() -> dict:
    if not os.path.exists(TIINGO_USAGE_FILE):
        return {'date': _tiingo_today(), 'calls': 0, 'tickers': {}}
    try:
        with open(TIINGO_USAGE_FILE, 'r', encoding='utf-8') as f:
            usage = json.load(f)
    except Exception:
        return {'date': _tiingo_today(), 'calls': 0, 'tickers': {}}
    if usage.get('date') != _tiingo_today():
        return {'date': _tiingo_today(), 'calls': 0, 'tickers': {}}
    usage.setdefault('calls', 0)
    usage.setdefault('tickers', {})
    return usage


def _save_tiingo_usage(usage: dict) -> None:
    Path(TIINGO_USAGE_FILE).parent.mkdir(parents=True, exist_ok=True)
    tmp = TIINGO_USAGE_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(usage, f, indent=2, sort_keys=True)
    os.replace(tmp, TIINGO_USAGE_FILE)


def _record_tiingo_call(ticker: str, status_code: int) -> None:
    usage = _load_tiingo_usage()
    usage['calls'] = int(usage.get('calls', 0)) + 1
    usage.setdefault('tickers', {})[ticker] = {
        'last_status': status_code,
        'last_attempt': pd.Timestamp.utcnow().isoformat(),
    }
    _save_tiingo_usage(usage)


def _tiingo_calls_used_today() -> int:
    return int(_load_tiingo_usage().get('calls', 0))


def _tiingo_run_budget(queue_size: int) -> int:
    daily_room = max(0, TIINGO_DAILY_LIMIT - TIINGO_RESERVE - _tiingo_calls_used_today())
    requested = queue_size if TIINGO_MAX_CALLS is None else max(0, TIINGO_MAX_CALLS)
    return max(0, min(queue_size, requested, daily_room))


def _select_tiingo_column(df: pd.DataFrame, *names: str) -> str | None:
    by_lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        found = by_lower.get(name.lower())
        if found is not None:
            return found
    return None


def _standardise_tiingo_frame(raw: pd.DataFrame,
                              output_ticker: str) -> pd.DataFrame:
    """
    Convert Tiingo's CSV payload to the project's adjusted OHLCV schema.

    Tiingo returns both raw and adjusted columns. The factor pipeline uses Yahoo
    auto-adjusted prices, so recovery rows prefer adjOpen/adjHigh/adjLow/
    adjClose/adjVolume and fall back to raw columns only when adjusted fields
    are unavailable.
    """
    if raw.empty:
        return EMPTY_PRICE_FRAME.copy()

    date_col = _select_tiingo_column(raw, 'date')
    close_col = _select_tiingo_column(raw, 'adjClose', 'close')
    if date_col is None or close_col is None:
        return EMPTY_PRICE_FRAME.copy()

    out = pd.DataFrame({
        'Date': raw[date_col],
        'Ticker': output_ticker,
        'Open': raw[_select_tiingo_column(raw, 'adjOpen', 'open')]
                if _select_tiingo_column(raw, 'adjOpen', 'open') else np.nan,
        'High': raw[_select_tiingo_column(raw, 'adjHigh', 'high')]
                if _select_tiingo_column(raw, 'adjHigh', 'high') else np.nan,
        'Low': raw[_select_tiingo_column(raw, 'adjLow', 'low')]
               if _select_tiingo_column(raw, 'adjLow', 'low') else np.nan,
        'Close': raw[close_col],
        'Volume': raw[_select_tiingo_column(raw, 'adjVolume', 'volume')]
                  if _select_tiingo_column(raw, 'adjVolume', 'volume') else np.nan,
    })
    return _standardise_frame(out, output_ticker)


def fetch_tiingo(ticker: str,
                 output_ticker: str | None = None) -> tuple[pd.DataFrame, str]:
    """Fetch one ticker's full daily history from Tiingo."""
    if not TIINGO_API_KEY:
        return EMPTY_PRICE_FRAME.copy(), 'missing Tiingo API key'

    url = TIINGO_PRICE_URL.format(ticker=ticker.lower())
    params = {
        'startDate': START_DATE[:10],
        'endDate': END_DATE[:10],
        'format': 'csv',
    }
    headers = {
        **HEADERS,
        'Authorization': f'Token {TIINGO_API_KEY}',
        'Accept': 'text/csv,application/json,text/plain',
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        _record_tiingo_call(ticker.upper(), r.status_code)
    except requests.exceptions.RequestException as exc:
        return EMPTY_PRICE_FRAME.copy(), f'network error: {exc}'

    body = r.text or ''
    if r.status_code == 401:
        return EMPTY_PRICE_FRAME.copy(), 'unauthorized token'
    if r.status_code == 404:
        return EMPTY_PRICE_FRAME.copy(), 'ticker not found'
    if r.status_code == 429:
        return EMPTY_PRICE_FRAME.copy(), 'Tiingo rate limit'
    if r.status_code != 200:
        return EMPTY_PRICE_FRAME.copy(), f'HTTP {r.status_code}'
    if not body.strip():
        return EMPTY_PRICE_FRAME.copy(), 'empty response'
    if body.lstrip().startswith('{') or body.lstrip().startswith('['):
        return EMPTY_PRICE_FRAME.copy(), 'non-CSV response'

    try:
        raw = pd.read_csv(StringIO(body))
    except Exception as exc:
        return EMPTY_PRICE_FRAME.copy(), f'CSV parse error: {exc}'

    df = _standardise_tiingo_frame(raw, output_ticker or ticker.upper())
    if df.empty:
        return EMPTY_PRICE_FRAME.copy(), 'no usable adjusted prices'
    return df, f'{len(df):,} rows'


def recover_missing_tickers(tickers: list[str],
                             existing_tickers: set[str],
                             alt_tickers: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Recover missing daily histories through Tiingo only.

    The function prints a plan before making requests, obeys the local daily
    quota ledger, and treats --dry-run/--tiingo-plan as zero-call modes.
    """
    already_done = existing_tickers if (RESUME and os.path.exists(DAILY_FILE)) else set()
    queue = [t for t in sorted(set(tickers)) if t not in already_done]
    budget = _tiingo_run_budget(len(queue))
    used = _tiingo_calls_used_today()

    print(f"\n[Recovery] Tiingo queue: {len(queue)} tickers")
    print(f"  Local Tiingo calls today: {used}/{TIINGO_DAILY_LIMIT} "
          f"(reserve: {TIINGO_RESERVE})")
    print(f"  Run call budget: {budget}")

    if not queue:
        return EMPTY_PRICE_FRAME.copy()
    if TIINGO_PLAN or DRY_RUN:
        sample = ', '.join(queue[:25])
        suffix = '...' if len(queue) > 25 else ''
        print(f"  Plan only: {sample}{suffix}")
        return EMPTY_PRICE_FRAME.copy()
    if not TIINGO_API_KEY:
        print("  Tiingo recovery skipped: set TIINGO_API_KEY before running.")
        return EMPTY_PRICE_FRAME.copy()
    if budget <= 0:
        print("  Tiingo recovery skipped: local quota guard has no calls available.")
        return EMPTY_PRICE_FRAME.copy()

    frames: list[pd.DataFrame] = []
    hits = 0
    failed = 0
    calls_started = 0

    for i, ticker in enumerate(queue, 1):
        if calls_started >= budget:
            remaining = len(queue) - i + 1
            print(f"  Quota guard stopped before {remaining} remaining tickers.")
            break

        label = f"[{i:>3}/{len(queue)}] {ticker:<8}"
        df, status = fetch_tiingo(ticker)
        calls_started += 1
        if not df.empty:
            print(f"  {label}  OK Tiingo ({status})")
            frames.append(df)
            hits += 1
            time.sleep(TIINGO_DELAY)
            continue

        alt = (alt_tickers or {}).get(ticker)
        if alt and calls_started < budget:
            alt_df, alt_status = fetch_tiingo(alt, output_ticker=ticker)
            calls_started += 1
            if not alt_df.empty:
                print(f"  {label}  OK Tiingo via {alt} ({alt_status})")
                frames.append(alt_df)
                hits += 1
                time.sleep(TIINGO_DELAY)
                continue
            status = f'{status}; {alt} fallback: {alt_status}'

        print(f"  {label}  missing ({status})")
        failed += 1
        time.sleep(TIINGO_DELAY)

    print(f"\n  Tiingo recovery: {hits}/{len(queue)} tickers recovered "
          f"| {failed} failed | {calls_started} calls used this run")

    if not frames:
        return EMPTY_PRICE_FRAME.copy()
    return pd.concat(frames, ignore_index=True)


def merge_recovered(recovered: pd.DataFrame) -> None:
    """Merge recovered rows into DAILY_FILE using keep='first' semantics:
    existing Yahoo rows win for data quality; recovered-only rows survive
    unchanged (no Yahoo row exists for them, so there is no conflict)."""
    if recovered.empty:
        return
    main = pd.read_csv(DAILY_FILE, parse_dates=['Date'])
    combined = (pd.concat([main, recovered], ignore_index=True)
                .drop_duplicates(subset=['Date', 'Ticker'], keep='first')
                .sort_values(['Ticker', 'Date']).reset_index(drop=True))
    tmp = DAILY_FILE + '.recovery_tmp'
    combined.to_csv(tmp, index=False)
    os.replace(tmp, DAILY_FILE)
    added = len(combined) - len(main)
    print(f"  Merged {added:,} new rows from recovery → {DAILY_FILE}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — OPTIONAL SEC EDGAR TERMINAL PRICE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

EDGAR_EFTS    = "https://efts.sec.gov/LATEST/search-index"
EDGAR_HEADERS = {**HEADERS, 'Accept': 'application/json, text/plain, */*'}

def _load_edgar_name_map(cache: str = 'edgar_cik_names.json') -> dict[str, str]:
    """Download SEC ticker→legal_name map once and cache to disk."""
    if Path(cache).exists():
        with open(cache) as f:
            return json.load(f)
    try:
        r = requests.get('https://www.sec.gov/files/company_tickers.json',
                         headers=EDGAR_HEADERS, timeout=30)
        if r.status_code != 200:
            return {}
        mapping = {}
        for v in r.json().values():
            ticker = str(v['ticker']).upper()
            # strip /DE/ /NV/ etc from end of legal name
            name = re.sub(r'\s*/\w{2,3}/\s*$', '', v['title'].strip())
            mapping[ticker] = name
        with open(cache, 'w') as f:
            json.dump(mapping, f)
        print(f"  [EDGAR] name map: {len(mapping):,} tickers cached → {cache}")
        return mapping
    except Exception as e:
        print(f"  [EDGAR] name map load failed: {e}")
        return {}

# Every pattern here requires explicit cash/merger/tender language.
# This excludes EPS ("$2.47 per share"), dividends ("$0.57 per share"),
# option grants, and CVR payouts — all common false positives in 8-K filings.
_ACQUISITION_PRICE_PATTERNS = [
    # "per share in cash" — the canonical cash deal phrase; EPS never says this
    r'\$\s*(\d{1,4}(?:\.\d{1,4})?)\s+per\s+share\s+in\s+cash',
    r'\$\s*(\d{1,4}(?:\.\d{1,4})?)\s+per\s+(?:common\s+)?share\s+of\s+cash',
    # "in cash … per share" (reversed word order common in proxy/SC TO-T language)
    r'in\s+cash[^.]{0,60}\$\s*(\d{1,4}(?:\.\d{1,4})?)\s+per\s+share',
    # Tender offer — dedicated merger documents; highest precision
    r'tender\s+offer\s+(?:purchase\s+)?price\s+of\s+\$\s*(\d{1,4}(?:\.\d{1,4})?)',
    r'offer\s+price\s+of\s+\$\s*(\d{1,4}(?:\.\d{1,4})?)\s+per\s+share',
    # Merger/acquisition consideration with explicit context
    r'(?:merger|acquisition|transaction)\s+consideration\s+of\s+\$\s*(\d{1,4}(?:\.\d{1,4})?)\s+per\s+share',
    r'consideration\s+of\s+\$\s*(\d{1,4}(?:\.\d{1,4})?)\s+per\s+share\s+in\s+cash',
    r'consideration\s+per\s+share\s+(?:in\s+cash\s+)?of\s+\$\s*(\d{1,4}(?:\.\d{1,4})?)',
    # Receive / will receive — only in cash context
    r'receive\s+\$\s*(\d{1,4}(?:\.\d{1,4})?)\s+in\s+cash\s+per\s+share',
    r'receive\s+\$\s*(\d{1,4}(?:\.\d{1,4})?)\s+per\s+share\s+in\s+cash',
    # Merger agreement purchase price
    r'purchase\s+price\s+of\s+\$\s*(\d{1,4}(?:\.\d{1,4})?)\s+per\s+(?:common\s+)?share',
    r'aggregate\s+(?:cash\s+)?consideration[^.]{0,80}\$\s*(\d{1,4}(?:\.\d{1,4})?)\s+per\s+share',
]


def _extract_per_share_price(text: str) -> float | None:
    """Extract acquisition per-share cash consideration from filing text."""
    candidates = []
    for pat in _ACQUISITION_PRICE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            try:
                price = float(m.group(1))
                if 0.5 <= price <= 1500:
                    candidates.append(price)
            except (ValueError, IndexError):
                pass
    if not candidates:
        return None
    return Counter(candidates).most_common(1)[0][0]


def _edgar_fetch_filing_text(cik: str, accession_no: str, doc_filename: str = '') -> str:
    """
    Fetch the primary document text from an EDGAR filing.
    If doc_filename is provided, build the direct filing URL.
    Otherwise, fall back to scraping the index page.
    CIK is stripped of leading zeros.
    """
    cik = cik.lstrip('0')
    acc_clean = accession_no.replace('-', '')
    if len(acc_clean) != 18:
        return ''

    if doc_filename:
        # Direct filing URL – works around SEC ix viewer
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{doc_filename}"
    else:
        # Legacy fallback: index page scraping
        acc_dashed = f"{acc_clean[:10]}-{acc_clean[10:12]}-{acc_clean[12:]}"
        index_url  = (f"https://www.sec.gov/Archives/edgar/data/"
                      f"{cik}/{acc_clean}/{acc_dashed}-index.htm")
        r = _safe_get(index_url, timeout=30)
        if r is None or r.status_code != 200:
            return ''
        soup = BeautifulSoup(r.text, 'html.parser')
        doc_url = None
        for link in soup.find_all('a', href=True):
            href = link['href']
            if (href.endswith('.htm') or href.endswith('.html')) and '-index' not in href:
                doc_url = ('https://www.sec.gov' + href if href.startswith('/') else href)
                break
        if doc_url is None:
            for link in soup.find_all('a', href=True):
                if link['href'].endswith('.txt') and 'complete' in link['href'].lower():
                    doc_url = 'https://www.sec.gov' + link['href']
                    break
        if doc_url is None:
            return ''

    r2 = _safe_get(doc_url, timeout=45)
    if r2 is None or r2.status_code != 200:
        return ''
    if r2.text.lstrip().startswith('<'):
        return BeautifulSoup(r2.text, 'html.parser').get_text(' ')
    return r2.text

def edgar_find_acquisition_price(ticker: str,
                                  approx_date: pd.Timestamp | None = None,
                                  legal_name: str = '') -> float | None:
    """
    Query SEC EDGAR EFTS for filings that may disclose cash merger
    consideration for a ticker. The search uses legal names when available and
    lets the downstream regex decide whether a returned filing is relevant.
    """
    if approx_date is not None:
        start = (approx_date - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        end   = (approx_date + pd.Timedelta(days=180)).strftime('%Y-%m-%d')
    else:
        start, end = '2000-01-01', '2025-12-31'

    # Require merger-specific language in the query so EFTS filters out
    # earnings 8-Ks (which dominate "per share" searches) before results
    # reach us.  SC TO-T (tender offer) and SC 13E-3 (going-private) are
    # listed first — EFTS ranks by form order when relevance is tied.
    params = {
        'q':         (f'"{legal_name if legal_name else ticker}" '
                      f'("per share in cash" OR "merger consideration" '
                      f'OR "tender offer price" OR "acquisition price")'),
        'forms':     'SC TO-T,SC 13E-3,DEFM14A,8-K',
        'dateRange': 'custom',
        'startdt':   start,
        'enddt':     end,
        '_source':   'ciks,adsh,entity_name,period_of_report,form_type',
    }
    try:
        r = requests.get(EDGAR_EFTS, params=params, headers=EDGAR_HEADERS, timeout=25)
    except Exception:
        return None

    if r is None or r.status_code != 200:
        return None

    try:
        data = r.json()
        hits = data.get('hits', {}).get('hits', [])
    except Exception:
        return None

    if not hits:
        return None

    for hit in hits[:6]:
        src       = hit.get('_source', {})
        entity_id = str((src.get('ciks') or [None])[0] or '').strip()
        accession = str(src.get('adsh', '')).strip()
        hit_id    = hit.get('_id', '')
        doc_file  = hit_id.split(':')[-1] if ':' in hit_id else ''

        if not entity_id or not accession:
            continue

        text = _edgar_fetch_filing_text(entity_id, accession, doc_file)
        if not text:
            time.sleep(0.5)
            continue

        price = _extract_per_share_price(text)
        if price is not None:
            return price

        time.sleep(0.8)

    return None


def run_edgar_recovery(missing_tickers: list[str],
                        daily_df: pd.DataFrame) -> dict[str, float]:
    """
    Run EDGAR acquisition price lookup for tickers that:
    1. Are still absent from daily_df, OR have very thin data
    2. Are NOT already in CURATED_TERMINAL
    3. Are NOT known to have zero cash consideration (spin-offs, bankruptcies)
    """
    skip_event_types = {'spin-off', 'bankruptcy', 'conservatorship'}
    already_known    = set(CURATED_TERMINAL.keys())
    tickers_in_csv   = set(daily_df['Ticker'].unique()) if not daily_df.empty else set()

    candidates = []
    for ticker in missing_tickers:
        if ticker in already_known:
            continue
        ct = CURATED_TERMINAL.get(ticker)
        if ct and ct[2] in skip_event_types:
            continue
        candidates.append(ticker)

    if not candidates:
        return {}

    print(f"\n[EDGAR] Searching SEC filings for {len(candidates)} tickers…")
    discovered: dict[str, float] = {}
    name_map = _load_edgar_name_map()

    for i, ticker in enumerate(candidates, 1):
        legal_name = name_map.get(ticker, '')
        label_name = f"{ticker}  ({legal_name})" if legal_name else ticker
        print(f"  [{i:>3}/{len(candidates)}] {label_name:<40}", end='  ', flush=True)
        approx_date = None
        if ticker in tickers_in_csv:
            last = daily_df[daily_df['Ticker'] == ticker]['Date'].max()
            approx_date = pd.Timestamp(last)

        try:
            price = edgar_find_acquisition_price(ticker, approx_date, legal_name)
        except Exception as exc:
            print(f"error: {exc}")
            price = None

        if price is not None:
            discovered[ticker] = price
            print(f"✓  ${price:.2f}/share (from 8-K filing)")
        else:
            print("—  no cash consideration found")

        time.sleep(1.0)

    print(f"\n  EDGAR recovery: {len(discovered)}/{len(candidates)} prices found")
    return discovered


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — OPTIONAL RENAME RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"


def openfigi_resolve_batch(tickers: list[str],
                            batch_size: int = 10) -> dict[str, str]:
    """
    Map old symbols to current symbols when the same legal entity continues
    under a new ticker. Starts with the curated same-entity rename map, then
    optionally queries OpenFIGI for any remaining symbols.

    Note: only renames where the same legal entity continues under a new ticker
    are useful here.  Acquisitions/mergers return the acquirer's ticker, which
    would substitute a different company's price history — handled instead by
    terminal price injection in Section 8.
    """
    rename_map: dict[str, str] = {}
    rename_map.update(KNOWN_RENAMES)

    unknowns = [t for t in tickers if t not in rename_map]
    if not unknowns:
        return rename_map

    print(f"\n[OpenFIGI] Resolving {len(unknowns)} possible renames…")
    for i in range(0, len(unknowns), batch_size):
        batch = unknowns[i:i + batch_size]
        payload = [{"idType": "TICKER", "idValue": t, "exchCode": "US"}
                   for t in batch]
        try:
            r = requests.post(OPENFIGI_URL,
                              json=payload,
                              headers={"Content-Type": "application/json"},
                              timeout=20)
            if r.status_code not in (200, 206):
                time.sleep(6.0)
                continue
            results = r.json()
            for j, result in enumerate(results):
                orig_ticker = batch[j]
                data        = result.get('data', [])
                if not data:
                    continue
                figi_ticker = data[0].get('ticker', '')
                if figi_ticker and figi_ticker != orig_ticker:
                    rename_map[orig_ticker] = figi_ticker
                    print(f"  {orig_ticker} → {figi_ticker}")
        except Exception:
            pass
        time.sleep(2.5)

    renamed_count = sum(1 for k, v in rename_map.items() if k != v and k in unknowns)
    print(f"  OpenFIGI: {renamed_count} new renames discovered")
    return rename_map


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TERMINAL PRICE INJECTION
# ══════════════════════════════════════════════════════════════════════════════

def _next_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts + pd.offsets.MonthBegin(1)
    if d.dayofweek == 5:
        d += pd.Timedelta(days=2)
    elif d.dayofweek == 6:
        d += pd.Timedelta(days=1)
    return d


def inject_terminal_prices(daily_df: pd.DataFrame,
                            extra_prices: dict[str, float] | None = None,
                            terminal_csv: str | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """
    Inject synthetic terminal price rows into the daily DataFrame.

    Sources (in priority order):
    1. CURATED_TERMINAL — hand-verified prices from SEC filings
    2. extra_prices     — prices discovered by EDGAR at runtime
    3. terminal_csv     — user-maintained terminal_prices.csv (if present)

    A synthetic row is only injected when:
    (a) The ticker has >= 1 existing row in daily_df (factor pipeline needs history)
    (b) The injection date is strictly AFTER the ticker's last existing row
    (c) The event is NOT a share-for-share merger (no cash → price is None)
    """
    all_prices: dict[str, tuple[float, str]] = {}
    for ticker, (price, date_str, _, _) in CURATED_TERMINAL.items():
        if price is not None:
            all_prices[ticker] = (price, date_str)
    if extra_prices:
        for ticker, price in extra_prices.items():
            if ticker not in all_prices:
                all_prices[ticker] = (price, '')
    if terminal_csv and os.path.exists(terminal_csv):
        tp = pd.read_csv(terminal_csv, parse_dates=['event_date', 'last_price_date'])
        for _, row in tp.iterrows():
            t = str(row['ticker']).upper().strip()
            if t not in all_prices:
                all_prices[t] = (float(row['last_price']),
                                 str(row['last_price_date'].date()))

    existing_tickers = set(daily_df['Ticker'].unique())
    log: list[dict] = []
    new_rows: list[pd.DataFrame] = []

    for ticker, (price, date_str) in all_prices.items():
        if ticker not in existing_tickers:
            log.append({'ticker': ticker, 'action': 'skipped_no_history',
                        'price': price, 'date': date_str})
            continue

        if date_str:
            try:
                base_date = pd.Timestamp(date_str)
            except Exception:
                base_date = daily_df[daily_df['Ticker'] == ticker]['Date'].max()
        else:
            base_date = daily_df[daily_df['Ticker'] == ticker]['Date'].max()

        injection_date = _next_business_day(base_date)
        last_existing  = pd.Timestamp(daily_df[daily_df['Ticker'] == ticker]['Date'].max())

        if injection_date <= last_existing:
            log.append({'ticker': ticker, 'action': 'skipped_already_covered',
                        'price': price, 'date': date_str})
            continue

        synthetic = pd.DataFrame([{'Date': injection_date, 'Ticker': ticker,
                                    'Open': price, 'High': price, 'Low': price,
                                    'Close': price, 'Volume': 0}])
        new_rows.append(synthetic)
        log.append({'ticker': ticker, 'action': 'injected',
                    'price': price, 'date': date_str,
                    'injection_date': injection_date.date()})

    if new_rows:
        combined = pd.concat([daily_df] + new_rows, ignore_index=True)
        combined = (combined.drop_duplicates(subset=['Date','Ticker'], keep='last')
                    .sort_values(['Ticker','Date']).reset_index(drop=True))
        n = sum(1 for d in log if d['action'] == 'injected')
        print(f"  ✓ Injected {n} synthetic terminal rows")
        return combined, log
    return daily_df, log


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — FACTOR CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_factors(daily_df: pd.DataFrame,
                  get_members,
                  current_tickers: set[str]) -> pd.DataFrame:
    """
    Build the monthly factor panel from daily OHLCV data.

    Factors:
      mom_6m_skip1  : 6-month momentum skip-1-month
      vol_1m        : 1-month realised volatility (annualised)
      vol_chg       : recent volume vs prior 3-month average
      fwd_ret       : one-month forward return
      fwd_ret_src   : 'natural' | 'terminal_recovered' (for bias study)
    """
    df = daily_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['log_ret'] = df.groupby('Ticker')['Close'].transform(
        lambda x: np.log(x / x.shift(1)))

    df['year_month'] = df['Date'].dt.to_period('M')
    me_idx  = df.groupby(['Ticker', 'year_month'])['Date'].idxmax()
    monthly = (df.loc[me_idx].copy()
               .sort_values(['Ticker', 'Date']).reset_index(drop=True))

    unique_dates = sorted(monthly['Date'].unique())
    mem_cache    = {d: get_members(d) for d in unique_dates}
    monthly['in_index'] = monthly.apply(
        lambda row: row['Ticker'] in mem_cache[row['Date']], axis=1)
    monthly = monthly[monthly['in_index']].copy()
    print(f"  After membership filter: {len(monthly):,} rows")

    monthly['close_lag1']   = monthly.groupby('Ticker')['Close'].shift(1)
    monthly['close_lag7']   = monthly.groupby('Ticker')['Close'].shift(7)
    monthly['mom_6m_skip1'] = monthly['close_lag1'] / monthly['close_lag7'] - 1

    df['vol_1m_daily'] = df.groupby('Ticker')['log_ret'].transform(
        lambda x: x.rolling(21, min_periods=5).std() * np.sqrt(252))
    monthly = monthly.merge(
        df[['Date', 'Ticker', 'vol_1m_daily']], on=['Date', 'Ticker'], how='left')
    monthly.rename(columns={'vol_1m_daily': 'vol_1m'}, inplace=True)

    df['vol_avg_21']       = df.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(21, min_periods=1).mean())
    df['vol_avg_63_prior'] = df.groupby('Ticker')['Volume'].transform(
        lambda x: x.shift(21).rolling(63, min_periods=1).mean())
    monthly = monthly.merge(
        df[['Date', 'Ticker', 'vol_avg_21', 'vol_avg_63_prior']],
        on=['Date', 'Ticker'], how='left')
    monthly['vol_chg'] = monthly['vol_avg_21'] / monthly['vol_avg_63_prior'] - 1

    monthly['close_next'] = monthly.groupby('Ticker')['Close'].shift(-1)
    monthly['fwd_ret']    = monthly['close_next'] / monthly['Close'] - 1
    monthly['fwd_ret_src'] = 'natural'

    print("  Computing terminal returns for last observations…")
    daily_by_ticker = {
        t: grp.set_index('Date')['Close'].sort_index()
        for t, grp in df.groupby('Ticker')}

    def _terminal_fwd(row):
        t, me = row['Ticker'], row['Date']
        nme   = me + pd.offsets.MonthEnd(1)
        if t not in daily_by_ticker:
            return np.nan
        window = daily_by_ticker[t].loc[(daily_by_ticker[t].index > me) &
                                        (daily_by_ticker[t].index <= nme)]
        if window.empty:
            return np.nan
        return float(window.iloc[-1] / row['Close'] - 1)

    last_idx  = monthly.groupby('Ticker').tail(1).index
    terminal  = monthly.loc[last_idx].apply(_terminal_fwd, axis=1)
    nan_mask  = monthly.loc[last_idx, 'fwd_ret'].isna()
    fill_idx  = nan_mask[nan_mask].index
    monthly.loc[fill_idx, 'fwd_ret']     = terminal.loc[fill_idx]
    monthly.loc[fill_idx, 'fwd_ret_src'] = 'terminal_recovered'
    n_filled = (~monthly.loc[fill_idx, 'fwd_ret'].isna()).sum()
    print(f"  Terminal returns recovered: {n_filled}/{len(fill_idx)}")

    monthly = monthly.dropna(subset=['fwd_ret', 'mom_6m_skip1', 'vol_1m', 'vol_chg'])

    all_hist = set().union(*[mem_cache[d] for d in unique_dates])
    in_panel  = set(monthly['Ticker'].unique())
    absent    = all_hist - in_panel
    pct_abs   = 100.0 * len(absent) / len(all_hist) if all_hist else 0
    if pct_abs > 10:
        print(f"\n  ⚠️  SURVIVORSHIP BIAS WARNING: {len(absent)}/{len(all_hist)} "
              f"({pct_abs:.1f}%) historical tickers absent from factor panel.")
    else:
        print(f"  Coverage: {len(absent)}/{len(all_hist)} ({pct_abs:.1f}%) absent "
              f"— below 10% warning threshold.")

    # Winsorise AFTER terminal fill — the terminal row (often a distressed price)
    # must participate in clipping so it anchors the cross-sectional tail correctly.
    monthly['fwd_ret'] = monthly.groupby('Date')['fwd_ret'].transform(
        lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)))

    cols = ['Date', 'Ticker', 'mom_6m_skip1', 'vol_1m', 'vol_chg',
            'fwd_ret', 'fwd_ret_src']
    return monthly[cols].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — BACKTEST + BIAS QUANTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _zscore_cross(grp: pd.DataFrame, col: str) -> pd.Series:
    mu, sigma = grp[col].mean(), grp[col].std()
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(0.0, index=grp.index)
    return (grp[col] - mu) / sigma


def _assign_deciles(grp: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    if grp['consensus'].dropna().__len__() < n_deciles:
        grp['decile'] = np.nan
        return grp
    try:
        grp['decile'] = (pd.qcut(grp['consensus'], n_deciles,
                                  labels=False, duplicates='drop') + 1)
    except Exception:
        grp['decile'] = np.nan
    return grp


def build_composite(df: pd.DataFrame,
                    mom_w: float = 1.0,
                    vol_w: float = -1.0,
                    vc_w:  float = 1.0,
                    n_deciles: int = 10) -> pd.DataFrame:
    df = df.copy()
    df['z_mom']    = df.groupby('Date', group_keys=False).apply(
        lambda g: _zscore_cross(g, 'mom_6m_skip1'))
    df['z_vol']    = df.groupby('Date', group_keys=False).apply(
        lambda g: _zscore_cross(g, 'vol_1m'))
    df['z_volchg'] = df.groupby('Date', group_keys=False).apply(
        lambda g: _zscore_cross(g, 'vol_chg'))
    # Composite score: high z_mom (momentum winner) + LOW z_vol (smooth, risk-parity
    # crowded) + high z_volchg (surging volume).  z_vol is subtracted so that LOW
    # volatility raises the consensus score — the risk-parity crowding thesis.
    df['consensus'] = mom_w * df['z_mom'] + vol_w * df['z_vol'] + vc_w * df['z_volchg']
    df = df.groupby('Date', group_keys=False).apply(
        lambda g: _assign_deciles(g, n_deciles))
    return df.dropna(subset=['decile'])


def _slice(s: pd.Series, start=None, end=None) -> pd.Series:
    if start: s = s[s.index >= start]
    if end:   s = s[s.index <= end]
    return s


def run_backtest(factor_df: pd.DataFrame,
                 mom_w: float = 1.0, vol_w: float = -1.0, vc_w: float = 1.0,
                 n_deciles: int = 10) -> dict:
    """Run the anti-consensus backtest; return performance metrics dict."""
    df = build_composite(factor_df, mom_w, vol_w, vc_w, n_deciles)
    df['decile'] = df['decile'].astype(int)

    long_r  = df[df['decile'] == 1].groupby('Date')['fwd_ret'].mean()
    short_r = df[df['decile'] == n_deciles].groupby('Date')['fwd_ret'].mean()
    ls_ret  = long_r - short_r

    # One-way turnover measures the fraction of last month's long book that is
    # exited this month, which is the quantity used by the cost model below.
    prev_long: set[str] | None = None
    turnovers: list[float] = []
    for d in sorted(long_r.index):
        curr = set(df[(df['decile'] == 1) & (df['Date'] == d)]['Ticker'])
        if prev_long is not None and len(prev_long) > 0:
            turnovers.append(len(prev_long - curr) / len(prev_long))
        prev_long = curr
    to_long = np.mean(turnovers) if turnovers else 0.50

    return {
        'long_r': long_r, 'short_r': short_r, 'ls_ret': ls_ret,
        'to_long': to_long,
        'full':  _perf(ls_ret, 'Full period'),
        'is':    _perf(_slice(ls_ret, end=IS_END), 'IS'),
        'oos':   _perf(_slice(ls_ret, start=OOS_START), '★ OOS'),
    }


# ── Bias Layer Quantification ─────────────────────────────────────────────────

def run_bias_study(factor_df_full: pd.DataFrame,
                   current_tickers: set[str],
                   daily_df: pd.DataFrame) -> None:
    """
    Quantify survivorship bias by running three backtest layers.

    Layer 0 is built by calling build_factors() with a naive get_members that
    always returns current_tickers for every date. This means:
    - Only current S&P 500 stocks ever enter the membership filter
    - A stock that joined the index in 2023 is counted as a member in 2005
    - A stock removed in 2008 never appears (survivorship bias fully active)
    The resulting factor_df_naive is what a naive analyst actually computes.

    Layer 0 — Naive baseline (maximum bias)
      Factor panel re-built with static current-only membership.
      Natural returns only.

    Layer 1 — Membership fixed
      Uses historical point-in-time membership.
      Natural returns only.

    Layer 2 — Full pipeline (this script)
      Historical membership + terminal return recovery + recovered tickers.

    Delta(0→1) = bias from using wrong (static) membership
    Delta(1→2) = bias from dropping terminal returns + missing data
    Delta(0→2) = total measurable survivorship bias
    """
    print("\n" + "═" * 72)
    print("  BIAS LAYER QUANTIFICATION")
    print("═" * 72)

    # Re-build the factor panel with naive constant membership
    # so that factor values themselves (momentum, vol) reflect the naive universe.
    def _naive_get_members(_as_of_date: pd.Timestamp) -> set[str]:
        return current_tickers

    print("\n  Building Layer 0 (naive, current-only) factor panel…")
    factor_df_naive = build_factors(daily_df, _naive_get_members, current_tickers)
    layer0_df = (factor_df_naive
                 .dropna(subset=['fwd_ret', 'mom_6m_skip1', 'vol_1m', 'vol_chg'])
                 .query('fwd_ret_src == "natural"')
                 .copy())

    # Layer 1: historical membership, natural returns only
    layer1_df = (factor_df_full
                 .dropna(subset=['fwd_ret', 'mom_6m_skip1', 'vol_1m', 'vol_chg'])
                 .query('fwd_ret_src == "natural"')
                 .copy())

    # Layer 2: full pipeline (already computed)
    layer2_df = (factor_df_full
                 .dropna(subset=['fwd_ret', 'mom_6m_skip1', 'vol_1m', 'vol_chg'])
                 .copy())

    layers = [
        ('Layer 0  Naive (static current membership)',  layer0_df),
        ('Layer 1  Historical membership, drop NaN',    layer1_df),
        ('Layer 2  Full pipeline (this script) ★',     layer2_df),
    ]

    results = []
    for label, df in layers:
        if len(df) < 100:
            print(f"  {label}: insufficient data ({len(df)} rows)")
            results.append(None)
            continue
        bt = run_backtest(df)
        results.append(bt)
        m_oos = bt['oos']
        m_is  = bt['is']
        print(f"\n  {label}")
        print(f"    Rows in panel : {len(df):,}")
        print(f"    IS  CAGR      : {m_is.get('cagr', np.nan):+.2%}  "
              f"Sharpe: {m_is.get('sharpe', np.nan):.2f}")
        print(f"    OOS CAGR      : {m_oos.get('cagr', np.nan):+.2%}  "
              f"Sharpe: {m_oos.get('sharpe', np.nan):.2f}  "
              f"MaxDD: {m_oos.get('max_dd', np.nan):.2%}")

    if all(r is not None for r in results):
        print("\n  " + "─" * 70)
        print("  BIAS DELTAS (positive = bias inflating strategy returns)")
        r0, r1, r2 = results
        print(f"\n  Membership bias  (Layer 0 − Layer 1):")
        print(f"    OOS CAGR   delta: {r0['oos'].get('cagr',0) - r1['oos'].get('cagr',0):+.2%}")
        print(f"    OOS Sharpe delta: {r0['oos'].get('sharpe',0) - r1['oos'].get('sharpe',0):+.2f}")
        print(f"\n  Terminal-return bias  (Layer 1 − Layer 2):")
        print(f"    OOS CAGR   delta: {r1['oos'].get('cagr',0) - r2['oos'].get('cagr',0):+.2%}")
        print(f"    OOS Sharpe delta: {r1['oos'].get('sharpe',0) - r2['oos'].get('sharpe',0):+.2f}")
        print(f"\n  Total measurable survivorship bias  (Layer 0 − Layer 2):")
        print(f"    OOS CAGR   delta: {r0['oos'].get('cagr',0) - r2['oos'].get('cagr',0):+.2%}")
        print(f"    OOS Sharpe delta: {r0['oos'].get('sharpe',0) - r2['oos'].get('sharpe',0):+.2f}")
        print("\n  NOTE: Remaining bias (unknown unknowns, pre-2005 membership")
        print("  errors, corporate-action adjustments) is not captured above.")
        print("  The true survivorship bias is >= the figures reported here.")

    if SAVE_CSV:
        rows = []
        for label, df in layers:
            if df is None or len(df) < 100:
                continue
            bt_row = run_backtest(df)
            rows.append({'layer': label,
                         'n_rows': len(df),
                         'oos_cagr': bt_row['oos'].get('cagr'),
                         'oos_sharpe': bt_row['oos'].get('sharpe'),
                         'oos_maxdd': bt_row['oos'].get('max_dd'),
                         'is_cagr': bt_row['is'].get('cagr'),
                         'is_sharpe': bt_row['is'].get('sharpe')})
        pd.DataFrame(rows).to_csv('bias_layers.csv', index=False)
        print("\n  Saved → bias_layers.csv")


def download_ff5() -> pd.DataFrame:
    """Download Kenneth French's 5-factor data (monthly, decimal format)."""
    try:
        r = requests.get(FF5_URL, timeout=30, headers=HEADERS)
        if r.status_code != 200:
            return pd.DataFrame()
        import io as _io
        with zipfile.ZipFile(_io.BytesIO(r.content)) as z:
            fname = [f for f in z.namelist()
                     if f.lower().endswith('.csv')][0]
            raw = z.read(fname).decode('latin-1')

        lines = raw.splitlines()
        start = next((i for i, l in enumerate(lines)
                      if re.match(r'\s*\d{6}', l)), None)
        if start is None:
            return pd.DataFrame()
        end = next((i for i in range(start, len(lines))
                    if lines[i].strip() == ''), len(lines))
        df = pd.read_csv(StringIO('\n'.join(lines[start:end])),
                         header=None,
                         names=['yyyymm','Mkt_RF','SMB','HML','RMW','CMA','RF'])
        df['Date'] = (pd.to_datetime(df['yyyymm'].astype(str), format='%Y%m')
                      + pd.offsets.MonthEnd(0))
        for col in ['Mkt_RF','SMB','HML','RMW','CMA','RF']:
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
        return df.set_index('Date').dropna()
    except Exception:
        return pd.DataFrame()


def ols_newey_west(y: np.ndarray, X: np.ndarray, n_lags: int = 4) -> dict:
    """
    OLS with Newey-West HAC standard errors using a Bartlett kernel.
    """
    n, k    = X.shape
    beta    = np.linalg.lstsq(X, y, rcond=None)[0]
    resid   = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)

    # Newey-West HAC covariance of the score vector
    scores = X * resid[:, None]
    S = scores.T @ scores
    for lag in range(1, n_lags + 1):
        w  = 1.0 - lag / (n_lags + 1)          # Bartlett kernel weight
        Sl = scores[lag:].T @ scores[:-lag]
        S += w * (Sl + Sl.T)

    V  = (n / max(n - k, 1)) * XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 1e-30))
    t  = beta / se

    if _HAS_SCIPY:
        p = 2.0 * _norm.sf(np.abs(t))
    else:
        # Abramowitz & Stegun 26.2.17 normal-CDF approximation.
        x_ = np.abs(t)
        a1, a2, a3, a4, a5 = (0.319381530, -0.356563782,
                               1.781477937, -1.821255978, 1.330274429)
        k_ = 1.0 / (1.0 + 0.2316419 * x_)
        kp = ((((a5 * k_ + a4) * k_ + a3) * k_ + a2) * k_ + a1) * k_
        tail = kp * np.exp(-0.5 * x_ * x_) / np.sqrt(2.0 * np.pi)
        p = np.minimum(1.0, 2.0 * tail)

    alpha_m   = float(beta[0])
    alpha_ann = (1.0 + alpha_m) ** 12 - 1.0
    return {'beta': beta, 'se': se, 't': t, 'p': p,
            'n': n, 'alpha_ann': alpha_ann}


def print_full_backtest_report(factor_df: pd.DataFrame) -> None:
    """
    Print the comprehensive backtest report.
    Sections: IS/OOS summary, FF5 alpha, sensitivity grid (IS only), all-decile spread.
    """
    def _header(s): print(f"\n{'═'*W}\n  {s}\n{'─'*W}")
    def _sub(s):    print(f"\n  ── {s}")

    _header("ANTI-CONSENSUS STRATEGY — FULL BACKTEST REPORT")
    print(f"  Factor file   : {FACTOR_FILE}")
    print(f"  IS period     : ≤ {IS_END.date()}")
    print(f"  OOS period    : ≥ {OOS_START.date()}  (★ headline numbers)")

    df = factor_df.sort_values(['Date', 'Ticker'])

    # ── §1: IS/OOS Long-Short ─────────────────────────────────────────────────
    _header("§1  LONG-SHORT  (Decile 1 vs Decile 10)")

    bt     = run_backtest(df)
    ls     = bt['ls_ret']
    ls_is  = _slice(ls, end=IS_END)
    ls_oos = _slice(ls, start=OOS_START)

    def _prow(m, label=''):
        print(f"  {label:<32}  "
              f"CAGR:{m.get('cagr', np.nan):>+8.2%}  "
              f"Sharpe:{m.get('sharpe', np.nan):>6.2f}  "
              f"MaxDD:{m.get('max_dd', np.nan):>8.2%}  "
              f"Hit:{m.get('hit', np.nan):>6.1%}")

    print()
    for tier_id, tier in COST_TIERS.items():
        rt  = tier['rt_bps'] / 10_000
        net = ls - bt['to_long'] * rt
        _prow(_perf(net), f"{tier['label']} (RT {tier['rt_bps']} bps)")
    print()
    _prow(bt['is'], "IS gross")
    _prow(bt['oos'], "★ OOS gross")

    # ── §2: FF5 Alpha ─────────────────────────────────────────────────────────
    _header("§2  FAMA-FRENCH 5-FACTOR ALPHA  (OOS, Newey-West HAC)")
    ff5 = pd.DataFrame()
    if not NO_FF:
        print("  Downloading FF5 factors…", end=' ', flush=True)
        ff5 = download_ff5()
        print("✓" if not ff5.empty else "✗")

    # Long-only decile-1 series
    df_comp = build_composite(df, 1.0, -1.0, 1.0, 10)
    lo_r    = df_comp[df_comp['decile'] == 1].groupby('Date')['fwd_ret'].mean()
    lo_net  = lo_r - bt['to_long'] * (LONG_COST_BPS_OW / 10_000) * 2

    if not ff5.empty:
        oos_common = lo_net.index.intersection(ff5.index)
        oos_common = oos_common[oos_common >= pd.Timestamp('2015-01-01')]
        if len(oos_common) >= 24:
            y  = (lo_net[oos_common] - ff5.loc[oos_common, 'RF']).values
            Xc = np.column_stack([
                np.ones(len(y)),
                ff5.loc[oos_common, ['Mkt_RF','SMB','HML','RMW','CMA']].values])
            res = ols_newey_west(y, Xc, n_lags=4)
            print(f"\n  Alpha (ann.)  : {res['alpha_ann']:+.2%}  "
                  f"t={res['t'][0]:.2f}  p={res['p'][0]:.3f}")
            for j, fname in enumerate(['Mkt-RF','SMB','HML','RMW','CMA'], 1):
                print(f"  {fname:<8} β={res['beta'][j]:+.3f}  "
                      f"t={res['t'][j]:.2f}  p={res['p'][j]:.3f}")
        else:
            print("  Insufficient OOS overlap with FF5 data.")

    # ── §3: Sensitivity grid (IS only) ────────────────────────────────────────
    _header("§3  SENSITIVITY GRID  (IS only — IS data ≤ 2014-12-31)")
    rows = []
    for spec_label, mw, vw, vcw in SIGNAL_SPECS:
        for nd in DECILE_COUNTS:
            is_df = df[df['Date'] <= IS_END].copy()
            if len(is_df) < 100:
                continue
            bt_s = run_backtest(is_df, mw, vw, vcw, nd)
            # Extract IS metrics only
            is_metrics = bt_s['is']
            rows.append({
                'spec': str(spec_label),
                'n_deciles': int(nd),
                'is_cagr':   is_metrics.get('cagr', np.nan),
                'is_sharpe': is_metrics.get('sharpe', np.nan),
            })
    grid = pd.DataFrame(rows)
    if not grid.empty:
        for nd in DECILE_COUNTS:
            g = grid[grid['n_deciles'] == nd]
            if g.empty:
                continue
            print(f"\n  n_deciles={nd}")
            print(f"  {'Spec':<34}  {'IS CAGR':>8}  {'IS SR':>7}")
            print('  ' + '─' * 55)
            for _, row in g.iterrows():
                star = '★' if '★' in str(row['spec']) else ' '
                is_cagr = f"{row['is_cagr']:>+8.2%}" if not pd.isna(row['is_cagr']) else "     N/A"
                is_sr   = f"{row['is_sharpe']:>7.2f}" if not pd.isna(row['is_sharpe']) else "   N/A"
                print(f"  {star}{row['spec']:<33}  {is_cagr}  {is_sr}")
    if SAVE_CSV and not grid.empty:
        grid.to_csv('sensitivity_grid.csv', index=False)

    # ── §4: All-decile spread ─────────────────────────────────────────────────
    _header("§4  ALL-DECILE SPREAD  (OOS, annualised)")
    df_oos = df[df['Date'] >= OOS_START].copy()
    if not df_oos.empty:
        dc = build_composite(df_oos, 1.0, -1.0, 1.0, 10)
        dc['decile'] = dc['decile'].astype(int)
        dec_rets = dc.groupby('decile')['fwd_ret'].mean() * 12
        print("  Note: Decile returns are arithmetic annualised (monthly mean × 12);"
              " the LS CAGR is geometric. Differences reflect volatility drag.")
        for dec, ret in dec_rets.items():
            tag = ' ← LONG' if dec == 1 else (' ← SHORT' if dec == 10 else '')
            print(f"  Decile {dec:>2}  {ret:>+8.2%}{tag}")

    print(f"\n{'═'*W}")
    print("  ★ OOS (2015→) figures are the only valid research claims.")
    print("  Do not re-tune after observing OOS results.")
    print(f"{'═'*W}\n")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def build_registry(all_missing: list[str],
                   daily_df: pd.DataFrame,
                   edgar_prices: dict[str, float],
                   injection_log: list[dict]) -> pd.DataFrame:
    """Build the per-ticker transparency log."""
    inj_map = {d['ticker']: d for d in injection_log}
    registry_df = daily_df.copy()
    if not registry_df.empty:
        registry_df['Date'] = pd.to_datetime(registry_df['Date'], errors='coerce')
        registry_df = registry_df.dropna(subset=['Date'])

    in_csv  = set(registry_df['Ticker'].unique()) if not registry_df.empty else set()
    ranges  = (registry_df.groupby('Ticker')['Date'].agg(['min','max'])
               .rename(columns={'min':'first_date','max':'last_date'})
               if not registry_df.empty else pd.DataFrame())

    def _date_or_none(value):
        ts = pd.to_datetime(value, errors='coerce')
        if pd.isna(ts):
            return None
        return pd.Timestamp(ts).date()

    rows = []
    for ticker in sorted(set(all_missing)):
        ct    = CURATED_TERMINAL.get(ticker)
        ev    = ct[2] if ct else 'unknown'
        inj   = inj_map.get(ticker, {})
        has_d = ticker in in_csv
        rng   = ranges.loc[ticker] if (has_d and ticker in ranges.index) else None

        rows.append({
            'ticker':                  ticker,
            'data_status':             'present' if has_d else 'absent',
            'first_date':              _date_or_none(rng['first_date']) if rng is not None else None,
            'last_date':               _date_or_none(rng['last_date'])  if rng is not None else None,
            'event_type':              ev,
            'terminal_injected':       inj.get('action') == 'injected',
            'terminal_price':          inj.get('price', ''),
            'terminal_price_source':   ct[3] if ct else (
                                       'EDGAR 8-K (dynamic)' if ticker in edgar_prices else ''),
            'edgar_price_found':       ticker in edgar_prices,
        })

    return (pd.DataFrame(rows)
            .sort_values(['data_status', 'ticker'])
            .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("  sp500_ceiling.py")
    print("=" * 72)
    if not DRY_RUN:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Data directory: {DATA_DIR}")
    # ── LIGHTNING MODE: skip all scraping & membership work ──
    LIGHTNING = '--lightning' in sys.argv
    if LIGHTNING:
        print("  ⚡ LIGHTNING MODE – using only the daily CSV, no downloads or membership DB")
        daily_df = pd.read_csv(DAILY_FILE, parse_dates=['Date'])
        current_tickers = set(daily_df['Ticker'].unique())
        get_members = lambda d: current_tickers
        yahoo_missing = []
        all_tickers_ever = current_tickers.copy()
        delisted_tickers = set()
        fja_membership = {}
        wiki_anchors = {}
        combined_anchors = {}
        unknown_unknowns = []
        all_missing = MISSING_TICKERS_HARDCODED
        still_absent = []
        edgar_prices = {}
        injection_log = []

        print("\n  Skipping Phases 1–5, going directly to Factor Construction.")    
    if not LIGHTNING:
        # ── PHASE 1: Membership Database ──────────────────────────────────────────
        print("\n" + "─" * 72)
        print("  PHASE 1 — Membership Database")
        print("─" * 72)

        membership_sets, current_tickers, history = build_membership_from_wikipedia()
        all_tickers_ever = set().union(*membership_sets.values())
        delisted_tickers = all_tickers_ever - current_tickers
        print(f"  Historical universe: {len(all_tickers_ever)} | "
            f"Active: {len(current_tickers)} | Delisted: {len(delisted_tickers)}")

        fja_membership = load_fja05680()

        target_years = list(range(2001, pd.Timestamp.today().year + 1))
        wiki_anchors = build_quarterly_wiki_anchors(target_years)

        combined_anchors: dict[pd.Timestamp, set[str]] = {}
        combined_anchors.update(fja_membership)
        combined_anchors.update(wiki_anchors)   # quarterly wiki overrides fja for same dates

        if wiki_anchors and not DRY_RUN:
            anchor_rows = [{'date': ts.date(), 'ticker_count': len(s),
                            'tickers': '|'.join(sorted(s))}
                        for ts, s in sorted(wiki_anchors.items())]
            pd.DataFrame(anchor_rows).to_csv(ANCHORS_FILE, index=False)
            print(f"  Quarterly anchors → {ANCHORS_FILE}")

        get_members = get_members_factory(membership_sets, combined_anchors)

        # ── PHASE 2: Price Download ────────────────────────────────────────────────
        print("\n" + "─" * 72)
        print("  PHASE 2 — Price Download")
        print("─" * 72)

        if (NO_DOWNLOAD or DRY_RUN) and os.path.exists(DAILY_FILE):
            mode = '--dry-run' if DRY_RUN else '--no-download'
            print(f"  {mode}: using existing {DAILY_FILE}")
            daily_df       = pd.read_csv(DAILY_FILE, parse_dates=['Date'])
            yahoo_missing  = []
        elif DRY_RUN:
            print("  --dry-run: no existing daily CSV found; skipping price download.")
            daily_df = pd.DataFrame()
            yahoo_missing = list(all_tickers_ever)
        elif NO_DOWNLOAD:
            print(f"  --no-download: {DAILY_FILE} not found.")
            daily_df = pd.DataFrame()
            yahoo_missing = list(all_tickers_ever)
        elif not _HAS_YF:
            print("  yfinance not installed — skipping download.")
            daily_df = (pd.read_csv(DAILY_FILE, parse_dates=['Date'])
                        if os.path.exists(DAILY_FILE) else pd.DataFrame())
            yahoo_missing = list(all_tickers_ever)
        else:
            daily_df, yahoo_missing = download_all_prices(current_tickers, delisted_tickers)

        tickers_in_csv = set(daily_df['Ticker'].unique()) if not daily_df.empty else set()

        # ── PHASE 3: Tiingo Recovery ──────────────────────────────────────────────
        print("\n" + "─" * 72)
        print("  PHASE 3 — Missing Ticker Recovery (Tiingo)")
        print("─" * 72)

        if not NO_RECOVERY:
            unknown_unknowns = discover_unknown_unknowns(
                all_tickers_ever, tickers_in_csv,
                MISSING_TICKERS_HARDCODED, daily_df)
            all_missing = list(set(MISSING_TICKERS_HARDCODED) |
                            set(unknown_unknowns) |
                            set(yahoo_missing))
            still_missing = [t for t in all_missing if t not in tickers_in_csv]

            rename_map = KNOWN_RENAMES.copy()
            if RUN_OPENFIGI:
                rename_map.update(openfigi_resolve_batch(still_missing))
            elif any(t in KNOWN_RENAMES for t in still_missing):
                known = {t: KNOWN_RENAMES[t] for t in still_missing if t in KNOWN_RENAMES}
                print(f"  Using curated rename map: {known}")

            recovered = recover_missing_tickers(
                still_missing, tickers_in_csv,
                alt_tickers={k: v for k, v in rename_map.items()
                            if k in still_missing and v != k})

            if not recovered.empty and not DRY_RUN:
                merge_recovered(recovered)
                daily_df       = pd.read_csv(DAILY_FILE, parse_dates=['Date'])
                tickers_in_csv = set(daily_df['Ticker'].unique())
        else:
            print("  --no-recovery: skipping.")
            unknown_unknowns = []
            all_missing      = MISSING_TICKERS_HARDCODED

        still_absent = [t for t in all_missing if t not in tickers_in_csv]

        # ── PHASE 4: Optional EDGAR Terminal Price Extraction ─────────────────────
        print("\n" + "─" * 72)
        print("  PHASE 4 — SEC EDGAR Terminal Price Extraction")
        print("─" * 72)

        edgar_prices: dict[str, float] = {}
        if RUN_EDGAR:
            edgar_prices = run_edgar_recovery(still_absent, daily_df)
        else:
            print("  Skipping EDGAR by default. Use --edgar to run it.")

        # ── PHASE 5: Terminal Price Injection ─────────────────────────────────────
        print("\n" + "─" * 72)
        print("  PHASE 5 — Terminal Price Injection")
        print("─" * 72)

        if daily_df.empty:
            print("  No daily data available — skipping injection.")
            injection_log = []
        else:
            updated_df, injection_log = inject_terminal_prices(
                daily_df, edgar_prices, TERMINAL_FILE)
            if not DRY_RUN:
                n_inj = sum(1 for d in injection_log if d['action'] == 'injected')
                if n_inj > 0:
                    tmp = DAILY_FILE + '.inject_tmp'
                    updated_df.to_csv(tmp, index=False)
                    os.replace(tmp, DAILY_FILE)
                    daily_df = updated_df
                    print(f"  {n_inj} terminal rows injected → {DAILY_FILE}")

    # ── PHASE 6: Factor Construction ──────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  PHASE 6 — Factor Construction")
    print("─" * 72)

    if daily_df.empty:
        print("  ERROR: no daily data — cannot build factors.")
        return

    print("[Factors] Building monthly factor panel…")
    factor_df = build_factors(daily_df, get_members, current_tickers)
    print(f"  {len(factor_df):,} observations | "
          f"{factor_df['Date'].min().date()} → {factor_df['Date'].max().date()}")

    if not DRY_RUN:
        factor_df.to_csv(FACTOR_FILE, index=False)
        print(f"  Saved → {FACTOR_FILE}")

    # ── PHASE 7: Backtest ──────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  PHASE 7 — Backtest")
    print("─" * 72)

    print_full_backtest_report(factor_df)

    if BIAS_STUDY:
        run_bias_study(factor_df, current_tickers, daily_df)

    # ── PHASE 8: Registry ─────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  PHASE 8 — Missing Ticker Registry")
    print("─" * 72)

    registry = build_registry(all_missing, daily_df, edgar_prices, injection_log)
    if not DRY_RUN:
        registry.to_csv(REGISTRY_FILE, index=False)
        print(f"  Registry → {REGISTRY_FILE}  ({len(registry)} tickers documented)")

    # ── Final Summary ─────────────────────────────────────────────────────────
    absent_count  = (registry['data_status'] == 'absent').sum()
    present_count = (registry['data_status'] == 'present').sum()
    injected      = registry['terminal_injected'].sum()
    edgar_count   = registry['edgar_price_found'].sum()

    print("\n" + "═" * 72)
    print("  FINAL SUMMARY")
    print("═" * 72)
    print(f"  Historical tickers tracked  : {len(all_tickers_ever)}")
    print(f"  Data present in CSV         : {present_count}")
    print(f"  Still absent (no free data) : {absent_count}")
    print(f"  Terminal prices injected    : {injected}")
    print(f"  EDGAR prices recovered      : {edgar_count}")
    print(f"  Missing/thin tickers found  : {len(unknown_unknowns)}")
    print(f"  Quarterly anchors loaded    : {len(wiki_anchors)}")
    print()
    print("  To quantify bias, re-run with --bias-study")
    print("  To save section CSVs, re-run with --csv")
    print("═" * 72)


if __name__ == '__main__':
    main()
