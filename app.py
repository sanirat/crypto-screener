import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

CG_BASE = "https://api.coingecko.com/api/v3"

# -----------------------------
# Polite request helper (handles 429 rate limits)
# -----------------------------
def cg_get(url: str, params: dict, max_retries: int = 7, base_sleep: float = 1.2):
    # Retries with exponential backoff on 429 (rate limit) and transient 5xx/network errors.
    backoff = base_sleep
    last_status = None
    for _ in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            last_status = r.status_code

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                time.sleep(sleep_s)
                backoff *= 1.8
                continue

            if 500 <= r.status_code < 600:
                time.sleep(backoff)
                backoff *= 1.6
                continue

            r.raise_for_status()
            return r.json()

        except requests.RequestException:
            time.sleep(backoff)
            backoff *= 1.6
            continue

    raise requests.HTTPError(f"CoinGecko request failed after retries (last status={last_status}).")


# -----------------------------
# Indicators
# -----------------------------
def rsi_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    if close.size < period + 2:
        return np.full(close.shape, np.nan, dtype=float)

    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    rsi = np.full(close.shape, np.nan, dtype=float)

    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    def rs(ag, al):
        if al == 0:
            return np.inf
        return ag / al

    rsi[period] = 100.0 - (100.0 / (1.0 + rs(avg_gain, avg_loss)))

    for i in range(period + 1, close.size):
        g = gain[i - 1]
        l = loss[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        rsi[i] = 100.0 - (100.0 / (1.0 + rs(avg_gain, avg_loss)))

    return rsi


def sma(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < n:
        return np.full(x.shape, np.nan, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def find_swings(series: np.ndarray, left: int = 2, right: int = 2, kind: str = "high"):
    s = np.asarray(series, dtype=float)
    idx = []
    for i in range(left, len(s) - right):
        wl = s[i - left:i]
        wr = s[i + 1:i + 1 + right]
        if kind == "high":
            if np.all(s[i] > wl) and np.all(s[i] > wr):
                idx.append(i)
        else:
            if np.all(s[i] < wl) and np.all(s[i] < wr):
                idx.append(i)
    return idx


def structure_pass(high: np.ndarray, low: np.ndarray, close: np.ndarray, struct_n: int = 60) -> bool:
    if len(close) < struct_n + 10:
        return False
    h = high[-struct_n:]
    l = low[-struct_n:]
    c = close[-struct_n:]

    hi_idx = find_swings(h, kind="high")
    lo_idx = find_swings(l, kind="low")

    if len(hi_idx) >= 2 and len(lo_idx) >= 2:
        h1, h2 = h[hi_idx[-2]], h[hi_idx[-1]]
        l1, l2 = l[lo_idx[-2]], l[lo_idx[-1]]
        return (h2 > h1) and (l2 > l1)

    s20 = sma(c, 20)
    s60 = sma(c, 60)
    if np.isnan(s20[-1]) or np.isnan(s60[-1]):
        return False
    return (c[-1] > s60[-1]) and (s20[-1] > s60[-1])


def is_probable_stablecoin(row) -> bool:
    sym = (row.get("symbol") or "").lower()
    name = (row.get("name") or "").lower()
    if "stable" in name:
        return True
    if any(x in sym for x in ["usdt", "usdc", "dai", "tusd", "fdusd", "busd", "usdd", "usde"]):
        return True
    return False


# -----------------------------
# CoinGecko helpers (cached)
# -----------------------------
@st.cache_data(ttl=60 * 30, show_spinner=False)
def cg_markets_page(page: int, per_page: int = 250):
    url = f"{CG_BASE}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": "false",
        "price_change_percentage": "30d",
    }
    return cg_get(url, params)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def cg_market_chart(coin_id: str, days: int = 370):
    url = f"{CG_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    return cg_get(url, params)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def cg_ohlc(coin_id: str, days: int = 90):
    url = f"{CG_BASE}/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    return cg_get(url, params)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Crypto Screener (50% off ATH + Bullish)", layout="wide")
st.title("Crypto screener: ≥50% off ATH + bullish (RSI + structure)")

with st.sidebar:
    st.header("Universe")
    max_pages = st.number_input("Market-cap pages to scan (250 coins/page)", min_value=1, max_value=40, value=4)
    min_mcap = st.number_input("Min market cap (USD)", min_value=0, value=200_000_000, step=50_000_000)
    min_vol = st.number_input("Min 24h volume (USD)", min_value=0, value=20_000_000, step=5_000_000)

    st.header("Rules")
    dd_threshold = st.slider("Drawdown from ATH (at least)", min_value=0.10, max_value=0.95, value=0.50, step=0.05)
    rsi_threshold = st.slider("RSI(14) minimum", min_value=40, max_value=70, value=50, step=1)
    struct_n = st.slider("Structure lookback (days)", min_value=30, max_value=120, value=60, step=5)

    st.header("Trend regime")
    allow_early_reversal = st.checkbox("Allow early reversal (near SMA200 + SMA50 rising)", value=True)

    st.header("CoinGecko limits")
    st.caption("If CoinGecko blocks requests, the app automatically waits and retries.")
    run = st.button("Run scan")

st.caption("Data source: CoinGecko public API. On shared hosting (Render Free), keep pages modest for stability.")

if not run:
    st.info("Adjust settings in the sidebar and click **Run scan**.")
    st.stop()

coins = []
progress = st.progress(0.0, text="Fetching market pages…")

for p in range(1, int(max_pages) + 1):
    try:
        coins.extend(cg_markets_page(p))
    except Exception as e:
        st.warning(
            "CoinGecko is blocking requests right now (rate limit). "
            "Try fewer pages, or wait 1–2 minutes and try again.\n\n"
            f"Details: {e}"
        )
        break
    progress.progress(p / max_pages, text=f"Fetched {p}/{max_pages} pages")
    time.sleep(0.15)

dfm = pd.DataFrame(coins)
if dfm.empty:
    st.error("No market data returned. If this keeps happening, CoinGecko may be blocking requests temporarily.")
    st.stop()

dfm = dfm[dfm["market_cap"].fillna(0) >= min_mcap]
dfm = dfm[dfm["total_volume"].fillna(0) >= min_vol]
dfm = dfm[~dfm.apply(is_probable_stablecoin, axis=1)]

dfm["drawdown"] = (dfm["current_price"] / dfm["ath"]) - 1.0
dfm = dfm[dfm["drawdown"] <= -dd_threshold].reset_index(drop=True)

st.write(f"After liquidity + stablecoin + drawdown filters: **{len(dfm)}** candidates.")

results = []
prog2 = st.progress(0.0, text="Computing RSI / structure / trend…")

for i, row in dfm.iterrows():
    coin_id = row["id"]
    try:
        mc = cg_market_chart(coin_id, days=370)
        prices = mc.get("prices", [])
        if len(prices) < 260:
            continue
        close = np.array([x[1] for x in prices], dtype=float)

        rsi14 = rsi_wilder(close, 14)
        rsi_now = float(rsi14[-1]) if not np.isnan(rsi14[-1]) else np.nan
        if np.isnan(rsi_now) or rsi_now < rsi_threshold:
            continue

        s50 = sma(close, 50)
        s200 = sma(close, 200)
        if np.isnan(s50[-1]) or np.isnan(s200[-1]):
            continue
        sma50_now = float(s50[-1])
        sma200_now = float(s200[-1])

        sma50_prev = float(s50[-11]) if len(s50) >= 11 and not np.isnan(s50[-11]) else np.nan
        sma50_rising = (not np.isnan(sma50_prev)) and (sma50_now > sma50_prev)

        regime_ok = (sma50_now > sma200_now)
        if (not regime_ok) and allow_early_reversal:
            regime_ok = sma50_rising and (close[-1] > sma200_now * 0.97)

        if not regime_ok:
            continue

        try:
            ohlc = cg_ohlc(coin_id, days=90)
            if isinstance(ohlc, list) and len(ohlc) >= struct_n:
                high = np.array([x[2] for x in ohlc], dtype=float)
                low = np.array([x[3] for x in ohlc], dtype=float)
                close_ohlc = np.array([x[4] for x in ohlc], dtype=float)
            else:
                high = close.copy()
                low = close.copy()
                close_ohlc = close.copy()
        except Exception:
            high = close.copy()
            low = close.copy()
            close_ohlc = close.copy()

        struct_ok = structure_pass(high, low, close_ohlc, struct_n=struct_n)
        if not struct_ok:
            continue

        ret30 = (close[-1] / close[-31] - 1.0) if len(close) >= 31 else np.nan

        score = 0.0
        score += 2.0 if (sma50_now > sma200_now) else 0.75
        score += 1.0 if (rsi_now >= 55) else 0.5
        score += 1.0
        score += 1.0 if (not np.isnan(ret30) and ret30 > 0) else 0.0
        score += 1.0 if (close[-1] > sma50_now) else 0.0

        results.append({
            "coin": row["name"],
            "symbol": row["symbol"].upper(),
            "price_usd": row["current_price"],
            "market_cap_usd": row["market_cap"],
            "volume_24h_usd": row["total_volume"],
            "ath_usd": row["ath"],
            "drawdown_pct": float(row["drawdown"] * 100.0),
            "rsi14": rsi_now,
            "ret30_pct": float(ret30 * 100.0) if not np.isnan(ret30) else np.nan,
            "sma50": sma50_now,
            "sma200": sma200_now,
            "sma_spread_pct": (sma50_now / sma200_now - 1.0) * 100.0,
            "score": score,
        })

    except Exception as e:
        st.warning(
            "CoinGecko rate-limited during the scan. Try fewer pages, or re-run in a minute.\n\n"
            f"Details: {e}"
        )
        break

    prog2.progress((i + 1) / max(1, len(dfm)), text=f"Processed {i+1}/{len(dfm)}")
    time.sleep(0.06)

if not results:
    st.warning("No matches under current settings. Try lowering RSI threshold, increasing pages, or relaxing liquidity filters.")
    st.stop()

out = pd.DataFrame(results).sort_values(["score", "market_cap_usd"], ascending=[False, False])

st.subheader("Matches")
st.dataframe(out, use_container_width=True)

st.download_button(
    "Download results as CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="crypto_screener_results.csv",
    mime="text/csv",
)
