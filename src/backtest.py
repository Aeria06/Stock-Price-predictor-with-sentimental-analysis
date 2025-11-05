
import pandas as pd
from .utils import metrics_regression

def walk_forward(X: pd.DataFrame, y: pd.Series, builder, n_windows=5, min_train=200):
    n = len(X)
    if n < (min_train + n_windows):
        n_windows = max(1, min(n_windows, n - min_train))
    window = (n - min_train) // n_windows if n_windows>0 else (n - min_train)
    preds, mets = [], []
    for i in range(n_windows):
        tr_end = min_train + i*window
        te_end = min(n, tr_end + window)
        X_tr, y_tr = X.iloc[:tr_end], y.iloc[:tr_end]
        X_te, y_te = X.iloc[tr_end:te_end], y.iloc[tr_end:te_end]
        if len(X_te)==0: break
        m = builder(); m.fit(X_tr, y_tr); y_pd = m.predict(X_te)
        preds.append((y_te, y_pd)); mets.append(metrics_regression(y_te, y_pd))
    y_all = pd.concat([p[0] for p in preds]) if preds else pd.Series(dtype=float)
    p_all = pd.concat([p[1] for p in preds]) if preds else pd.Series(dtype=float)
    overall = metrics_regression(y_all, p_all) if len(y_all) else {}
    return {"y_test_all": y_all, "y_pred_all": p_all, "per_split_metrics": mets, "overall": overall}
