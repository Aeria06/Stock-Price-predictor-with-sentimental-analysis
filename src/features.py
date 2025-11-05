
import pandas as pd, numpy as np
from .utils import ensure_1d_series

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    series = ensure_1d_series(series).astype(float)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def build_features(df: pd.DataFrame,
                   sma_windows=(5,10,20,50),
                   ema_windows=(12,26),
                   rsi_window=14,
                   vol_window=10,
                   news_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    if "Close" not in out: raise KeyError("'Close' column not found")
    close = ensure_1d_series(out["Close"], index=out.index, name="Close").astype(float)
    out["Close"] = close

    for w in sma_windows:
        out[f"SMA_{w}"] = close.rolling(w).mean()
    for w in ema_windows:
        out[f"EMA_{w}"] = close.ewm(span=w, adjust=False).mean()
    out[f"RSI_{rsi_window}"] = rsi(close, window=rsi_window)

    out["Return_1d"] = close.pct_change()
    out[f"Volatility_{vol_window}"] = out["Return_1d"].rolling(vol_window).std()

    if news_daily is not None and len(news_daily):
        nd = news_daily.copy()
        if not isinstance(nd.index, pd.DatetimeIndex):
            nd.index = pd.to_datetime(nd.index)
        nd = nd.sort_index()
        keep = [c for c in ["SentimentRaw","SentimentEMA_3"] if c in nd.columns]
        nd = nd[keep] if keep else pd.DataFrame(index=nd.index)
        out = out.join(nd, how="left")
    if "SentimentRaw" not in out: out["SentimentRaw"] = 0.0
    if "SentimentEMA_3" not in out: out["SentimentEMA_3"] = 0.0
    out["SentimentRaw"] = pd.to_numeric(out["SentimentRaw"], errors="coerce").astype(float).ffill().fillna(0.0)
    out["SentimentEMA_3"] = pd.to_numeric(out["SentimentEMA_3"], errors="coerce").astype(float).ffill().fillna(0.0)

    out["Close_t+1"] = close.shift(-1)

    # Drop duplicate columns (keep the last occurrence)
    out = out.loc[:, ~out.columns.duplicated(keep="last")]

    # Defensive: ensure the Close column is strictly 1-D before returning.
    arr = np.asarray(out["Close"])  # could be Series, ndarray, or (n,1) array
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.squeeze(1)
    if arr.ndim != 1:
        # Provide a clearer error describing the shape/type to help debugging
        raise AssertionError(
            f"'Close' must be 1-D after processing, got array with shape {arr.shape} and type {type(out['Close'])}."
            " Ensure you're passing a DataFrame with a single 'Close' column or a Series."
        )
    out["Close"] = pd.Series(arr, index=out.index, name="Close")
    return out.dropna()
