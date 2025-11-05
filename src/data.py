
import pandas as pd, numpy as np, datetime as dt
try:
    import yfinance as yf
except Exception:
    yf = None

def download_prices(ticker: str, period="2y", interval="1d") -> pd.DataFrame:
    if yf is not None:
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.rename(columns=str.title)
                df = df[['Open','High','Low','Close','Volume']].dropna()
                for c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
                return df
        except Exception:
            pass
    np.random.seed(42)
    n = 600
    dates = pd.date_range(end=dt.date.today(), periods=n, freq='B')
    steps = np.random.normal(loc=0.03, scale=1.1, size=n).cumsum()
    close = (100 + steps - steps.min()*0.1).clip(1.0)
    close = pd.Series(close, index=dates, name="Close")
    high = close * (1 + np.random.uniform(0, 0.02, size=n))
    low  = close * (1 - np.random.uniform(0, 0.02, size=n))
    open_= close.shift(1).fillna(close.iloc[0]) * (1 + np.random.uniform(-0.01, 0.01, size=n))
    vol  = np.random.randint(1e5, 6e5, size=n)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=dates)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df.dropna()
