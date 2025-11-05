
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def ensure_1d_series(x, index=None, name=None):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:,0]
        else:
            raise ValueError(f"Expected single column, got {x.shape[1]} columns")
    arr = np.asarray(x)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            arr = arr.squeeze(1)
        else:
            raise ValueError(f"Expected 1-D or (n,1), got {arr.shape}")
    return pd.Series(arr, index=(x.index if hasattr(x,'index') else index), name=name)

def metrics_regression(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = ensure_1d_series(y_true).astype(float)
    y_pred = ensure_1d_series(y_pred, index=y_true.index).astype(float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    ad = np.sign(y_true.diff()); pd_ = np.sign(y_pred.diff())
    idx = ad.dropna().index.intersection(pd_.dropna().index)
    da = float((ad.loc[idx]==pd_.loc[idx]).mean()) if len(idx) else float('nan')
    return {"MAE": mae, "RMSE": rmse, "Directional_Accuracy": da}

def time_split(df: pd.DataFrame, test_frac: float):
    split = int(len(df)*(1-test_frac))
    return df.iloc[:split].copy(), df.iloc[split:].copy()
