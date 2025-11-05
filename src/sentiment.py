
import pandas as pd, numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
_an = SentimentIntensityAnalyzer()

def score_text(t: str) -> float:
    if not isinstance(t, str) or not t.strip():
        return 0.0
    return float(_an.polarity_scores(t)["compound"])

def build_daily_sentiment(news_df: pd.DataFrame, date_col="date", text_col="text") -> pd.DataFrame:
    if news_df is None or news_df.empty: return pd.DataFrame(columns=["SentimentRaw","SentimentEMA_3"])
    df = news_df.copy()
    df = df.dropna(subset=[date_col, text_col])
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize()
    df["score"] = df[text_col].apply(score_text)
    daily = df.groupby(date_col)["score"].mean().to_frame("SentimentRaw")
    daily["SentimentEMA_3"] = daily["SentimentRaw"].ewm(span=3, adjust=False).mean()
    return daily.sort_index()
