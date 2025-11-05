import os, sys
# Ensure project root is on sys.path so the top-level `src` package can be
# imported when running this script from the `app/` directory (e.g. via
# `streamlit run app/app.py`). We insert the project root at the front of
# sys.path to prefer the local package.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt
from src.data import download_prices
from src.features import build_features
from src.sentiment import build_daily_sentiment
from src.utils import metrics_regression, time_split
from src.models import NaiveLag, LinReg, RFReg
from src.backtest import walk_forward

st.set_page_config(page_title="Stock Predict — SAFE", page_icon="🛡️", layout="wide")
st.title("🛡️ Stock Predict — SAFE Edition")

st.sidebar.header("Data")
ticker = st.sidebar.text_input("Ticker", value="TCS.NS")
period = st.sidebar.selectbox("Period", ["1y","2y","5y","10y","max"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1wk"], index=0)

st.sidebar.header("Model")
model_name = st.sidebar.selectbox("Primary Model", ["naive","linreg","rf"], index=2)
n_windows = st.sidebar.slider("Walk-forward windows", 1, 10, 5, 1)

st.sidebar.header("News / Sentiment")
mode = st.sidebar.radio("Add sentiment via", ["None","Paste text","Upload CSV"], index=0)
news_daily = None

with st.expander("How sentiment works"):
    st.write("Paste headlines/paragraphs (one per line) or upload CSV with columns `date,text`. We'll compute daily sentiment and add features `SentimentRaw` & `SentimentEMA_3`.")

if mode == "Paste text":
    txt = st.text_area("Paste news (one per line):", height=120)
    days_span = st.number_input("Apply to last N trading days", 1, 30, 3, 1)
elif mode == "Upload CSV":
    csv = st.file_uploader("CSV with columns: date,text", type=["csv"])

@st.cache_data(show_spinner=False)
def load_prices(ticker, period, interval):
    return download_prices(ticker, period, interval)

prices = load_prices(ticker, period, interval)

if mode == "Paste text" and txt and txt.strip():
    from src.sentiment import score_text
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    avg = float(np.mean([score_text(l) for l in lines])) if lines else 0.0
    idx = prices.index[-int(days_span):]
    news_daily = pd.DataFrame({"SentimentRaw":[avg]*len(idx)}, index=idx)
    news_daily["SentimentEMA_3"] = news_daily["SentimentRaw"].ewm(span=3, adjust=False).mean()
    st.sidebar.info(f"Avg sentiment: {avg:.3f}")
elif mode == "Upload CSV" and csv is not None:
    raw = pd.read_csv(csv)
    news_daily = build_daily_sentiment(raw, date_col="date", text_col="text")
    st.sidebar.success(f"Loaded {len(raw)} rows -> {len(news_daily)} daily points")

feat = build_features(prices, news_daily=news_daily)
# Defensive: ensure no duplicate columns (keep last occurrence) are passed to models
feat = feat.loc[:, ~feat.columns.duplicated(keep="last")]
y = feat["Close_t+1"]; X = feat.drop(columns=["Close_t+1"])

st.subheader("Raw Data (tail)"); st.dataframe(prices.tail(15))

train_df, test_df = time_split(feat, 0.2)
X_tr, y_tr = train_df.drop(columns=["Close_t+1"]), train_df["Close_t+1"]
X_te, y_te = test_df.drop(columns=["Close_t+1"]), test_df["Close_t+1"]

MODELS = {"naive": lambda: NaiveLag(), "linreg": lambda: LinReg(), "rf": lambda: RFReg()}
model = MODELS[model_name](); model.fit(X_tr, y_tr); y_pd = model.predict(X_te)
m = metrics_regression(y_te, y_pd)

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{m['MAE']:.2f}"); c2.metric("RMSE", f"{m['RMSE']:.2f}")
c3.metric("Directional Acc", f"{m['Directional_Accuracy']*100:.1f}%")

st.subheader("Actual vs Predicted — Holdout")
fig1 = plt.figure(figsize=(10,4))
plt.plot(y_te.index, y_te.values, label="Actual")
plt.plot(y_pd.index, y_pd.values, label="Predicted")
plt.legend(); plt.xlabel("Date"); plt.ylabel("Price"); plt.title("Holdout")
st.pyplot(fig1, clear_figure=True)

st.subheader("Walk-Forward Backtest")
from src.backtest import walk_forward
wf = walk_forward(X, y, MODELS[model_name], n_windows=n_windows, min_train=200)
m2 = wf["overall"]
c1, c2, c3 = st.columns(3)
c1.metric("WF MAE", f"{m2.get('MAE', float('nan')):.2f}" if m2 else "N/A")
c2.metric("WF RMSE", f"{m2.get('RMSE', float('nan')):.2f}" if m2 else "N/A")
c3.metric("WF Directional", f"{m2.get('Directional_Accuracy', float('nan'))*100:.1f}%" if m2 else "N/A")
fig2 = plt.figure(figsize=(10,4))
if len(wf["y_test_all"]):
    plt.plot(wf["y_test_all"].index, wf["y_test_all"].values, label="Actual")
    plt.plot(wf["y_pred_all"].index, wf["y_pred_all"].values, label="Predicted")
    plt.legend(); plt.xlabel("Date"); plt.ylabel("Price"); plt.title("Walk-Forward")
st.pyplot(fig2, clear_figure=True)

st.subheader("🏆 Leaderboard")
def eval_model(name):
    m_ = MODELS[name](); m_.fit(X_tr, y_tr); p_ = m_.predict(X_te)
    met = metrics_regression(y_te, p_); return {"Model": name, **met}
lb = pd.DataFrame([eval_model(n) for n in MODELS.keys()])
metric_choice = st.selectbox("Sort by", ["MAE","RMSE","Directional_Accuracy"], index=0)
asc = True if metric_choice in ("MAE","RMSE") else False
lb_ = lb.sort_values(metric_choice, ascending=asc).reset_index(drop=True)
st.dataframe(lb_)
fig3 = plt.figure(figsize=(6,4))
plt.bar(lb_["Model"], lb_[metric_choice].values); plt.title(f"Leaderboard — {metric_choice}")
st.pyplot(fig3, clear_figure=True)

st.subheader("Predictions (Holdout)")
outdf = pd.DataFrame({"Actual": y_te, "Predicted": y_pd})
st.dataframe(outdf.tail(25))
st.download_button("Download CSV", data=outdf.to_csv().encode("utf-8"), file_name=f"{ticker}_predictions.csv", mime="text/csv")

if model_name == "rf" and hasattr(model, "feature_importances"):
    st.subheader("RF Feature Importances")
    imp = model.feature_importances(X_tr.columns).sort_values(ascending=False)[:20]
    fig4 = plt.figure(figsize=(8,6))
    plt.barh(imp.index[::-1], imp.values[::-1]); plt.xlabel("Importance"); plt.title("Top Features")
    st.pyplot(fig4, clear_figure=True)

st.caption("Safe Edition: indicators are pure pandas; strict 1-D coercion throughout; synthetic fallback if Yahoo blocked.")
