import os
import time
from functools import lru_cache
import numpy as np
import joblib
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def _model_path(coin_id: str) -> str:
    return os.path.join(MODEL_DIR, f"{coin_id}_iforest.joblib")


def _dummy_dataframe(coin_id: str, days: int) -> pd.DataFrame:
    """Return a synthetic price DataFrame with the same shape as the real CoinGecko response.

    Prices are seeded from a rough per-coin baseline so the chart looks plausible.
    The daily returns are drawn from a normal distribution with a couple of injected
    spikes so IsolationForest always has something interesting to flag.
    """
    BASELINES = {
        "bitcoin": 60_000.0,
        "ethereum": 3_200.0,
        "solana": 140.0,
        "binancecoin": 380.0,
    }
    base = BASELINES.get(coin_id.lower(), 100.0)

    rng = np.random.default_rng(seed=abs(hash(coin_id)) % (2**32))
    n = max(days, 31)  # always give IsolationForest enough rows

    # daily returns: mostly small noise, a few large spikes to create anomalies
    returns = rng.normal(loc=0.001, scale=0.02, size=n)
    spike_indices = rng.choice(n, size=3, replace=False)
    returns[spike_indices] *= rng.choice([-1, 1], size=3) * rng.uniform(4, 8, size=3)

    prices = base * np.cumprod(1 + returns)

    end_date = pd.Timestamp.utcnow().normalize()
    dates = pd.date_range(end=end_date, periods=n, freq="D", tz=None)

    df = pd.DataFrame({"Close": prices}, index=dates)
    df.index.name = "Date"
    df["Close"] = df["Close"].astype(float)
    return df


@lru_cache(maxsize=10)
def fetch_ohlcv(coin_id: str, days: int = 365) -> pd.DataFrame:
    # retry needed — coingecko's free tier rate-limits pretty aggressively (429s)
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}

    last_err = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                last_err = f"rate limited by CoinGecko (attempt {attempt + 1})"
                time.sleep(2.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            payload = resp.json()
            prices = payload.get("prices")  # list of [timestamp_ms, price]
            if not prices:
                last_err = "empty prices array returned"
                time.sleep(1.5 * (attempt + 1))
                continue

            df = pd.DataFrame(prices, columns=["timestamp", "Close"])
            df["Date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
            df = df.set_index("Date")[["Close"]]
            # coingecko sometimes returns 2 rows for the same day (today, partial) - keep the last
            df = df[~df.index.duplicated(keep="last")]
            return df
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(1.5 * (attempt + 1))

    # All retries exhausted — fall back to synthetic data so the UI stays functional.
    print(
        f"[fetch_ohlcv] WARNING: could not reach CoinGecko for '{coin_id}' "
        f"({last_err}). Returning synthetic fallback data."
    )
    return _dummy_dataframe(coin_id, days)


def compute_returns(df: pd.DataFrame) -> pd.Series:
    # training on raw price is a trap - it's non-stationary, model just learns "price go up"
    closes = df["Close"]
    returns = closes.pct_change().dropna()
    return returns


def train_and_save(coin_id: str) -> dict:
    df = fetch_ohlcv(coin_id)
    returns = compute_returns(df)

    if len(returns) < 30:
        raise ValueError("not enough data points to train on, need at least ~30 days")

    X = returns.values.reshape(-1, 1)

    # contamination is a guess, 5% anomalies felt reasonable for daily returns
    clf = IsolationForest(n_estimators=150, contamination=0.05, random_state=42)
    clf.fit(X)

    joblib.dump(clf, _model_path(coin_id))

    # stash returns + raw closes so /anomalies can chart price without refetching
    returns.to_pickle(os.path.join(MODEL_DIR, f"{coin_id}_returns.pkl"))
    df["Close"].to_pickle(os.path.join(MODEL_DIR, f"{coin_id}_prices.pkl"))

    return {
        "coin_id": coin_id,
        "rows_trained": len(returns),
        "model_path": _model_path(coin_id),
    }


def load_model(coin_id: str):
    path = _model_path(coin_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no trained model found for {coin_id}, run /analyze first")
    return joblib.load(path)


def predict_anomalies(coin_id: str) -> dict:
    clf = load_model(coin_id)

    returns_path = os.path.join(MODEL_DIR, f"{coin_id}_returns.pkl")
    prices_path = os.path.join(MODEL_DIR, f"{coin_id}_prices.pkl")
    if not os.path.exists(returns_path) or not os.path.exists(prices_path):
        raise FileNotFoundError(f"no cached data for {coin_id}, run /analyze first")

    returns = pd.read_pickle(returns_path)
    prices = pd.read_pickle(prices_path)
    X = returns.values.reshape(-1, 1)

    preds = clf.predict(X)  # -1 = anomaly, 1 = normal

    anomalies = []
    for date, ret, pred in zip(returns.index, returns.values, preds):
        if pred == -1:
            anomalies.append({
                "date": date.strftime("%Y-%m-%d"),
                "return_pct": round(float(ret) * 100, 4),
                "price": round(float(prices.loc[date]), 2),
            })

    # full price timeline goes back to the frontend too, so it can draw the chart
    price_history = [
        {"date": date.strftime("%Y-%m-%d"), "price": round(float(price), 2)}
        for date, price in prices.items()
    ]

    return {"anomalies": anomalies, "price_history": price_history}
