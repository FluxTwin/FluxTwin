import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def _ensure_daily(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    s = df.set_index("timestamp")["consumption_kwh"].resample("D").sum().dropna()
    return s

def daily_forecast(df: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    """Return dataframe with columns: date, forecast_kwh, method."""
    s = _ensure_daily(df)
    if len(s) < 10:
        # Naive mean if λίγα δεδομένα
        mean = float(s.mean()) if len(s) else 0.0
        idx = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        return pd.DataFrame({"date": idx, "forecast_kwh": [mean]*horizon_days, "method": "naive-mean"})
    try:
        model = ExponentialSmoothing(s, trend="add", seasonal=None).fit()
        f = model.forecast(horizon_days)
        return pd.DataFrame({"date": f.index, "forecast_kwh": f.values, "method": "holt-winters"})
    except Exception:
        mean = float(s.mean())
        idx = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        return pd.DataFrame({"date": idx, "forecast_kwh": [mean]*horizon_days, "method": "naive-fallback"})
