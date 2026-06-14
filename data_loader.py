import io
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

CACHE_FILE = "tunisian_stocks_30y.csv"
FALLBACK_TICKERS = [
    'SFBT', 'BT', 'BIAT', 'SAH', 'PGH', 'STB', 'ATB', 'BH', 'BNA',
    'UIB', 'DH', 'DELICE', 'OTH', 'SOTUVER',
]
EXCLUDED_TICKERS = {'TBIDX', 'PX1', 'TVAL', 'TBIDXT'}
SCRAPE_CHUNK_DAYS = 90
OVERLAP_DAYS = 7  # re-fetch a few days to fill gaps after incremental updates


def _session_and_headers():
    session = requests.Session()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    return session, headers


def _fetch_ticker_list(session, headers):
    print("Fetching list of all active stock tickers from ilboursa.com...")
    try:
        r_aaz = session.get('https://www.ilboursa.com/marches/aaz', headers=headers, timeout=30)
        soup = BeautifulSoup(r_aaz.text, 'html.parser')
        tickers = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('cotation_'):
                tickers.append(href.split('_')[-1])
        tickers = sorted(set(tickers))
        tickers = [t for t in tickers if t not in EXCLUDED_TICKERS]
        print(f"Successfully found {len(tickers)} active tickers on the exchange.")
        return tickers
    except Exception as e:
        print(f"Error fetching ticker list: {e}. Falling back to major list.")
        return FALLBACK_TICKERS


def _normalize_ticker_df(ticker_df, ticker=None):
    ticker_df = ticker_df.copy()
    ticker_df.rename(columns={
        'symbole': 'Ticker',
        'date': 'Date',
        'ouverture': 'Open',
        'haut': 'High',
        'bas': 'Low',
        'cloture': 'Close',
        'volume': 'Volume',
    }, inplace=True)

    if ticker is not None and 'Ticker' not in ticker_df.columns:
        ticker_df['Ticker'] = ticker

    for col in ['Open', 'High', 'Low', 'Close']:
        if col in ticker_df.columns:
            ticker_df[col] = (
                ticker_df[col].astype(str).str.replace(',', '.').astype(float)
            )

    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'], format='%d/%m/%Y', errors='coerce')
    ticker_df = ticker_df.dropna(subset=['Date'])
    return ticker_df


def _scrape_ticker_range(session, headers, ticker, range_start, range_end):
    """Download OHLCV for one ticker between range_start and range_end (inclusive)."""
    download_url = f'https://www.ilboursa.com/marches/download/{ticker}'
    r_get = session.get(download_url, headers=headers, timeout=30)
    if r_get.status_code != 200:
        return None

    soup = BeautifulSoup(r_get.text, 'html.parser')
    token_input = soup.find('input', {'name': '__RequestVerificationToken'})
    if not token_input:
        return None
    token = token_input['value']

    ticker_dfs = []
    curr_end = range_end
    consecutive_empty = 0

    while curr_end > range_start:
        curr_start = max(range_start, curr_end - timedelta(days=SCRAPE_CHUNK_DAYS))
        payload = {
            'dtFrom': curr_start.strftime('%Y-%m-%d'),
            'dtTo': curr_end.strftime('%Y-%m-%d'),
            '__RequestVerificationToken': token,
        }
        time.sleep(0.1)
        r_post = session.post(download_url, data=payload, headers=headers, timeout=60)

        if 'text/csv' in r_post.headers.get('Content-Type', ''):
            df_chunk = pd.read_csv(io.StringIO(r_post.text), sep=';')
            if not df_chunk.empty:
                ticker_dfs.append(df_chunk)
                consecutive_empty = 0
            else:
                consecutive_empty += 1
        else:
            consecutive_empty += 1

        if consecutive_empty >= 3:
            break

        curr_end = curr_start - timedelta(days=1)

    if not ticker_dfs:
        return None

    ticker_df = pd.concat(ticker_dfs, ignore_index=True)
    return _normalize_ticker_df(ticker_df, ticker=ticker)


def _merge_frames(frames):
    if not frames:
        raise ValueError("Could not collect any historical stock data from the website.")
    final_df = pd.concat(frames, ignore_index=True)
    final_df = (
        final_df
        .drop_duplicates(subset=['Ticker', 'Date'], keep='last')
        .sort_values(['Ticker', 'Date'])
        .reset_index(drop=True)
    )
    return final_df


def _full_scrape(session, headers, tickers, start_date, end_date, cache_file):
    print(f"Full download: {start_date.date()} → {end_date.date()} for {len(tickers)} tickers...")
    full_df_list = []
    for ticker in tqdm(tickers, desc="Scraping stocks"):
        try:
            ticker_df = _scrape_ticker_range(session, headers, ticker, start_date, end_date)
            if ticker_df is not None and not ticker_df.empty:
                full_df_list.append(ticker_df)
        except Exception:
            pass

    final_df = _merge_frames(full_df_list)
    final_df.to_csv(cache_file, index=False)
    print(
        f"\nCached {len(final_df)} rows, "
        f"{final_df['Ticker'].nunique()} tickers, "
        f"through {final_df['Date'].max().date()} → {cache_file}"
    )
    return final_df


def _incremental_update(session, headers, df_cached, tickers, start_date, end_date, cache_file):
    """Extend cached data through end_date; full history for tickers not yet in cache."""
    max_cached = df_cached['Date'].max()
    print(
        f"Incremental update: cache ends {max_cached.date()}, "
        f"fetching through {end_date.date()}..."
    )

    update_start = max(start_date, max_cached - timedelta(days=OVERLAP_DAYS))
    cached_tickers = set(df_cached['Ticker'].unique())
    updated_frames = []

    for ticker in tqdm(tickers, desc="Updating stocks"):
        try:
            if ticker in cached_tickers:
                ticker_start = update_start
            else:
                ticker_start = start_date

            ticker_df = _scrape_ticker_range(
                session, headers, ticker, ticker_start, end_date,
            )
            if ticker_df is not None and not ticker_df.empty:
                if ticker in cached_tickers:
                    old = df_cached[df_cached['Ticker'] == ticker]
                    ticker_df = pd.concat([old, ticker_df], ignore_index=True)
                updated_frames.append(ticker_df)
            elif ticker in cached_tickers:
                updated_frames.append(df_cached[df_cached['Ticker'] == ticker])
        except Exception:
            if ticker in cached_tickers:
                updated_frames.append(df_cached[df_cached['Ticker'] == ticker])

    # Tickers removed from the exchange listing but still in cache
    for ticker in cached_tickers - set(tickers):
        updated_frames.append(df_cached[df_cached['Ticker'] == ticker])

    final_df = _merge_frames(updated_frames)
    final_df.to_csv(cache_file, index=False)
    print(
        f"\nUpdated cache: {len(final_df)} rows, "
        f"{final_df['Ticker'].nunique()} tickers, "
        f"through {final_df['Date'].max().date()}"
    )
    return final_df


def load_tunisian_stocks(years=30, force_refresh=False):
    """
    Load BVMT-listed Tunisian stock OHLCV from ilboursa.com.

    Uses a local CSV cache. On each run, if the cache is older than today,
    downloads new data through the current date (incremental when possible).
    Pass force_refresh=True (or env STOCKS_FORCE_REFRESH=1) for a full rebuild.
    """
    cache_file = CACHE_FILE
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    env_force = os.environ.get('STOCKS_FORCE_REFRESH', '').lower() in ('1', 'true', 'yes')
    force_refresh = force_refresh or env_force

    session, headers = _session_and_headers()
    tickers = _fetch_ticker_list(session, headers)

    if force_refresh and os.path.exists(cache_file):
        print("Force refresh requested — removing existing cache.")
        os.remove(cache_file)

    df_cached = None
    if os.path.exists(cache_file) and not force_refresh:
        try:
            df_cached = pd.read_csv(cache_file)
            df_cached['Date'] = pd.to_datetime(df_cached['Date'])
            n_tickers = df_cached['Ticker'].nunique()
            max_date = df_cached['Date'].max()

            if n_tickers <= 10:
                print("Old cached dataset with few tickers — rebuilding...")
                os.remove(cache_file)
                df_cached = None
            elif max_date.date() >= end_date.date():
                print(
                    f"Cache is current through {max_date.date()} "
                    f"({n_tickers} tickers, {len(df_cached)} rows)."
                )
                return df_cached
            else:
                return _incremental_update(
                    session, headers, df_cached, tickers,
                    start_date, end_date, cache_file,
                )
        except Exception as e:
            print(f"Error reading cache ({e}). Rebuilding...")
            if os.path.exists(cache_file):
                os.remove(cache_file)
            df_cached = None

    return _full_scrape(session, headers, tickers, start_date, end_date, cache_file)


# Alias for callers that refer to the exchange (BVMT) rather than the data vendor.
load_bvmt_stocks = load_tunisian_stocks
