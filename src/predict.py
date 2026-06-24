"""
predict.py — Sales Forecasting

Purpose:
    Load the LSTM model and forecast the next N days given the last
    LOOKBACK days of sales values.
"""

from __future__ import annotations

import os
import numpy as np

from tf_compat import require_tensorflow, tf

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "lstm_model.keras")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LOOKBACK = 30


def load_model_and_scale():
    require_tensorflow()
    model = tf.keras.models.load_model(MODEL_PATH)
    mn, mx = np.load(os.path.join(PROCESSED_DIR, "scale.npy"))
    return model, float(mn), float(mx)


def forecast_next_n_days(recent_sales: np.ndarray, n_days: int = 7,
                          model=None, mn: float | None = None, mx: float | None = None) -> np.ndarray:
    """
    Forecast the next n_days given the last LOOKBACK days of actual sales.

    Args:
        recent_sales: array of shape (LOOKBACK,) with the most recent sales
        n_days: how many future days to forecast
        model, mn, mx: optional pre-loaded artifacts

    Returns:
        np.ndarray of shape (n_days,) with forecasted sales values
    """
    if model is None:
        model, mn, mx = load_model_and_scale()

    # Normalize using training scale
    recent_norm = (recent_sales - mn) / (mx - mn)

    preds_norm = []
    window = list(recent_norm[-LOOKBACK:])

    for _ in range(n_days):
        x = np.array(window[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        pred = model.predict(x, verbose=0)[0][0]
        preds_norm.append(pred)
        window.append(pred)  # autoregressive: feed prediction back as input

    preds = np.array(preds_norm) * (mx - mn) + mn
    return preds


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "daily_sales.csv"),
                     parse_dates=["date"])
    recent = df["sales"].values[-LOOKBACK:]
    model, mn, mx = load_model_and_scale()
    forecast = forecast_next_n_days(recent, n_days=7, model=model, mn=mn, mx=mx)
    print("7-day forecast:")
    for i, v in enumerate(forecast, 1):
        print(f"  Day +{i}: {v:.1f}")
