# Daily Sales Forecasting — LSTM vs ARIMA

An end-to-end time-series forecasting pipeline that compares a classical
ARIMA model against an LSTM neural network — deployed as an interactive
Streamlit dashboard with adjustable forecast horizon.

![Architecture](images/architecture_diagram.png)

## Problem Statement

Retail and e-commerce businesses need accurate sales forecasts to manage
inventory, staffing, and marketing spend. Over- or under-forecasting
directly causes either stockouts (lost revenue) or excess inventory (wasted
capital). This project demonstrates a principled approach: start with a
classical ARIMA baseline, then compare against a deep learning LSTM.

## Dataset

730 days (2 years) of synthetic daily sales data with:
- Long-term upward **trend** (₹200 → ₹450 average)
- **Yearly seasonality** (summer peak, winter dip)
- **Weekly seasonality** (weekend vs weekday patterns)
- **Gaussian noise** (realistic day-to-day variability)

This same pipeline works on any real sales CSV with `date` and `sales` columns.

## Approach

1. **Preprocessing** (`src/data_preprocessing.py`): Calendar feature
   engineering (day of week, month, day of year), **min-max normalization
   fitted on training set only**, sliding-window sequences (`lookback=30`)
   for supervised LSTM training
2. **ARIMA baseline** (`src/train_model.py`): ARIMA(5,1,0) fitted on raw
   training data — 5 AR terms capture weekly seasonality, 1 differencing
   removes the linear trend
3. **LSTM model**: `LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)`
   trained with `EarlyStopping` on MSE loss — captures non-linear patterns
   the ARIMA model misses
4. **Autoregressive forecasting** (`src/predict.py`): feeds each prediction
   back as input for multi-step ahead forecasting
5. **Deployment**: Streamlit dashboard with adjustable forecast horizon,
   historical overlay, and model comparison chart

## Results (60-day test set)

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| ARIMA(5,1,0) | 62.63 | 68.40 | 14.92% |
| **LSTM** | **17.87** | **22.79** | **4.32%** |

![Forecast Comparison](images/forecast_comparison.png)
![Training Curves](images/training_curves.png)

LSTM outperforms ARIMA by ~3.5× on MAE by capturing the non-linear trend
and interactions between weekly/yearly cycles.

## Project Structure

```
sales-forecaster/
├── data/
│   ├── raw/daily_sales.csv
│   └── processed/
├── notebooks/eda.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── models/lstm_model.keras
├── app/streamlit_app.py
├── images/
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone https://github.com/<your-username>/sales-forecaster.git
cd sales-forecaster
pip install -r requirements.txt
python src/data_preprocessing.py
python src/train_model.py
python -m streamlit run app/streamlit_app.py
```

## Troubleshooting

If you see `ModuleNotFoundError: No module named 'tensorflow'`, the active
Python environment does not have this project's dependencies installed.
From the `sales-forecaster` directory, run:

```bash
pip install -r requirements.txt
```

Then rerun the script or Streamlit app from the same environment.

## Tech Stack

Python · TensorFlow/Keras (LSTM) · Statsmodels (ARIMA) · Pandas · NumPy
· Matplotlib · Streamlit

## Future Improvements

- Add Facebook Prophet for comparison (handles holidays and multiple
  seasonalities natively with minimal tuning)
- Implement multi-step direct forecasting instead of autoregressive
  (separate model per horizon — avoids error accumulation)
- Add confidence intervals to the forecast plot
