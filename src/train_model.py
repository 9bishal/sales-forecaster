"""
train_model.py — Sales Forecasting

Purpose:
    Train two models and compare them:
    1. ARIMA (classical, interpretable baseline)
    2. LSTM (deep learning, captures non-linear patterns)

    A classic interview move: always compare a simple baseline against
    your main model. If ARIMA (3 parameters) performs as well as LSTM,
    use ARIMA in production — simpler, faster, and easier to explain to
    stakeholders.

Run:
    python src/train_model.py
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from tf_compat import require_tensorflow, tf

warnings.filterwarnings("ignore")

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "lstm_model.keras")
PLOT_PATH     = os.path.join(PROJECT_ROOT, "images", "forecast_comparison.png")
HISTORY_PATH  = os.path.join(PROJECT_ROOT, "images", "training_curves.png")


def build_lstm(lookback: int = 30) -> tf.keras.Model:
    """
    Architecture: LSTM(64) -> Dropout(0.2) -> Dense(32) -> Dense(1)

    Why LSTM for time series?
    - LSTM cells have a 'memory' (cell state and hidden state) that lets
      them learn long-range dependencies in sequences — e.g., that a dip
      30 days ago tends to be followed by a recovery.
    - Unlike a simple Dense layer on the lookback window (which treats all
      days as independent features), LSTM processes the sequence step by
      step and maintains context.

    Why only 1 LSTM layer, not stacked?
    - Stacked LSTMs add capacity but also training time and overfitting
      risk. For a clean portfolio project, one LSTM layer is enough to
      demonstrate the concept and perform well on structured sales data.
      "I'd try stacking if the single-layer model underfits" is a strong
      interview answer.

    Why dropout after LSTM?
    - LSTM is prone to overfitting on small-to-medium datasets. Dropout
      randomly zeroes out units during training, forcing the network to
      learn redundant representations — a standard regularisation technique.
    """
    require_tensorflow()
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(lookback, 1), return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def denormalize(values: np.ndarray, mn: float, mx: float) -> np.ndarray:
    """Reverse min-max normalization back to original sales units."""
    return values * (mx - mn) + mn


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """
    Compute MAE, RMSE, and MAPE for time-series evaluation.

    Why these metrics instead of accuracy?
    - This is a regression problem — we predict a continuous sales value,
      not a class label.
    - MAE (Mean Absolute Error): average absolute difference in sales
      units — easy to explain to a business stakeholder ("off by ₹X on
      average").
    - RMSE (Root Mean Square Error): penalises large errors more than MAE.
      A large RMSE vs small MAE means there are occasional big misses.
    - MAPE (Mean Absolute Percentage Error): expresses error as a %,
      making it comparable across products with different sales scales.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    print(f"\n{name}:  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")
    return {"model": name, "mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}


def run_training():
    require_tensorflow()
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    mn, mx  = np.load(os.path.join(PROCESSED_DIR, "scale.npy"))
    test_raw = np.load(os.path.join(PROCESSED_DIR, "test_raw.npy"))

    # ------------------------------------------------------------------ #
    # Model 1: ARIMA baseline (classical)
    # ------------------------------------------------------------------ #
    # Use the full training series in denormalized form for ARIMA
    # (ARIMA works natively on the original scale)
    train_raw = denormalize(y_train, mn, mx)
    # Fit ARIMA(5,1,0): 5 autoregressive terms, 1 differencing, 0 MA terms
    # p=5 captures ~weekly seasonality; d=1 removes linear trend
    print("Fitting ARIMA(5,1,0)...")
    arima_model = ARIMA(train_raw, order=(5, 1, 0))
    arima_fit   = arima_model.fit()
    arima_preds = arima_fit.forecast(steps=len(y_test))
    arima_metrics = evaluate(test_raw[30:], arima_preds, "ARIMA(5,1,0)")

    # ------------------------------------------------------------------ #
    # Model 2: LSTM
    # ------------------------------------------------------------------ #
    print("\nTraining LSTM...")
    lstm_model = build_lstm(lookback=X_train.shape[1])
    history = lstm_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=8, restore_best_weights=True, monitor="val_loss"
            )
        ],
        verbose=0,
    )
    print(f"LSTM trained for {len(history.history['loss'])} epochs.")

    lstm_preds_norm = lstm_model.predict(X_test, verbose=0).flatten()
    lstm_preds      = denormalize(lstm_preds_norm, mn, mx)
    y_test_denorm   = denormalize(y_test, mn, mx)
    lstm_metrics    = evaluate(y_test_denorm, lstm_preds, "LSTM")

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("LSTM Loss (MSE)"); axes[0].legend()
    axes[1].plot(history.history["mae"], label="Train")
    axes[1].plot(history.history["val_mae"], label="Val")
    axes[1].set_title("LSTM MAE"); axes[1].legend()
    plt.tight_layout(); os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True); plt.savefig(HISTORY_PATH, dpi=120); plt.close()

    # Forecast comparison
    actual = y_test_denorm
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(actual,      label="Actual Sales",        color="black",   linewidth=1.5)
    ax.plot(lstm_preds,  label=f"LSTM  (MAE={lstm_metrics['mae']:.1f})",
            color="#1f77b4", linewidth=1.2)
    ax.plot(arima_preds, label=f"ARIMA (MAE={arima_metrics['mae']:.1f})",
            color="#d62728", linewidth=1.2, linestyle="--")
    ax.set_title("Sales Forecast: LSTM vs ARIMA (Test Set)")
    ax.set_xlabel("Day"); ax.set_ylabel("Sales")
    ax.legend()
    plt.tight_layout(); os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True); plt.savefig(PLOT_PATH, dpi=120); plt.close()

    # ------------------------------------------------------------------ #
    # Save LSTM model
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    lstm_model.save(MODEL_PATH)
    print(f"\nLSTM model saved to {MODEL_PATH}")
    return lstm_model, arima_metrics, lstm_metrics


if __name__ == "__main__":
    run_training()
