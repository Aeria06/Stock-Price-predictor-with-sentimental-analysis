
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class NaiveLag:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self
    def predict(self, X: pd.DataFrame):
        """Return a 1-D series of naive predictions.

        Behavior:
        - If a column named 'Close' exists (possibly duplicated), take the LAST occurrence.
        - Coerce the result to a 1-D ndarray and return a pd.Series with the same index.
        - Fall back to the last column if no 'Close' exists.
        """
        # If 'Close' present, selecting via loc may return a DataFrame when there are
        # duplicate column names. Handle both Series and DataFrame cases.
        if "Close" in X.columns:
            col = X.loc[:, "Close"]
            # DataFrame when duplicates exist
            if hasattr(col, "columns"):
                ser = col.iloc[:, -1]
            else:
                ser = col
            arr = np.asarray(ser).reshape(-1)
            return pd.Series(arr, index=X.index, name="NaivePred")
        # fallback: last column
        ser = X.iloc[:, -1]
        arr = np.asarray(ser).reshape(-1)
        return pd.Series(arr, index=X.index, name="NaivePred")

class LinReg:
    def __init__(self):
        self.m = LinearRegression()
    def fit(self, X, y):
        self.m.fit(X, y); return self
    def predict(self, X):
        return pd.Series(self.m.predict(X), index=X.index)

class RFReg:
    def __init__(self, n_estimators=400, random_state=42):
        self.m = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    def fit(self, X, y):
        self.m.fit(X, y); return self
    def predict(self, X):
        return pd.Series(self.m.predict(X), index=X.index)
    def feature_importances(self, cols):
        import pandas as pd
        # Coerce column labels to strings so downstream plotting (matplotlib)
        # which expects string/bytes labels does not fail when labels are tuples
        # (e.g. from accidental MultiIndex or tuple column names).
        idx = pd.Index(cols).map(lambda x: x if isinstance(x, (str, bytes)) else str(x))
        return pd.Series(self.m.feature_importances_, index=idx)
