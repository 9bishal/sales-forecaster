"""
data_preprocessing.py — Sales Forecasting

Purpose:
    Load raw daily sales, add calendar features, create the sliding-window
    sequences needed for the LSTM, and split into train/test sets.

Run:
    python src/data_preprocessing.py
"""

import os
import shutil
import numpy as np
import pandas as pd

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH       = os.path.join(PROJECT_ROOT, "data", "raw", "daily_sales.csv")
PROCESSED_DIR  = os.path.join(PROJECT_ROOT, "data", "processed")

LOOKBACK  = 30   # use the past 30 days to predict the next day
TEST_DAYS = 90   # hold out the last 90 days as the test set


def reset_processed_dir() -> None:
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_and_feature_engineer(path: str = RAW_PATH) -> pd.DataFrame:
    """
    Load the CSV and add simple calendar features.

    Features added:
    - day_of_week (0=Mon, 6=Sun): weekly seasonality signal
    - month (1-12): yearly seasonality signal
    - day_of_year (1-365): smooth yearly cycle

    Why calendar features?
    - Sales data almost always has weekly and yearly patterns.
      Calendar features give the model explicit access to these patterns
      rather than requiring it to infer them purely from the lagged values.
    - They're simple to create and explain in interviews.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"]       = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    return df


def normalize(train_series: np.ndarray, test_series: np.ndarray):
    """
    Min-max normalize using ONLY the training set's min/max.

    Why min-max and not StandardScaler here?
    - LSTMs work better when inputs are bounded in [0, 1]; min-max
      guarantees this bound for values within the training range.
    - StandardScaler could produce negative values and values outside
      [-1, 1], which interact less cleanly with the tanh/sigmoid
      activations in LSTM cells.
    - Fitted on training data only to prevent leakage of test statistics.
    """
    mn, mx = train_series.min(), train_series.max()
    train_norm = (train_series - mn) / (mx - mn)
    test_norm  = (test_series  - mn) / (mx - mn)
    return train_norm, test_norm, mn, mx


def make_sequences(series: np.ndarray, lookback: int = LOOKBACK):
    """
    Create sliding-window sequences for supervised learning.

    For a time series [a, b, c, d, e] with lookback=3:
      X = [[a,b,c], [b,c,d]]
      y = [d, e]

    This turns the forecasting problem into a supervised regression
    problem — the model learns: given the last 30 days of sales, predict
    tomorrow's sales.
    """
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i: i + lookback])
        y.append(series[i + lookback])
    return np.array(X), np.array(y)


def run_pipeline():
    reset_processed_dir()

    df = load_and_feature_engineer()
    print(f"Loaded {len(df)} rows: {df.date.min().date()} → {df.date.max().date()}")

    sales = df["sales"].values

    train_raw = sales[:-TEST_DAYS]
    test_raw  = sales[-TEST_DAYS:]

    train_norm, test_norm, mn, mx = normalize(train_raw, test_raw)

    X_train, y_train = make_sequences(train_norm, LOOKBACK)
    X_test,  y_test  = make_sequences(test_norm,  LOOKBACK)

    # LSTM expects shape: (samples, timesteps, features) — add feature dim
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test)
    np.save(os.path.join(PROCESSED_DIR, "scale.npy"), np.array([mn, mx]))

    # Also save the raw test sales for denormalized plotting
    np.save(os.path.join(PROCESSED_DIR, "test_raw.npy"), test_raw)
    df.to_csv(os.path.join(PROCESSED_DIR, "sales_featured.csv"), index=False)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Scale: min={mn:.1f}, max={mx:.1f}")
    return X_train, X_test, y_train, y_test, mn, mx


if __name__ == "__main__":
    run_pipeline()
