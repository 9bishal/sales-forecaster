"""
streamlit_app.py — Sales Forecasting Dashboard

Run: streamlit run app/streamlit_app.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# pyrefly: ignore [missing-import]
from predict import load_model_and_scale, forecast_next_n_days  # noqa

st.set_page_config(page_title="Sales Forecaster", layout="wide")
st.title("Daily Sales Forecasting — LSTM vs ARIMA")
st.write("Visualize historical sales and forecast the next N days using an LSTM model.")

st.info(
    "Trained on 2 years of synthetic daily sales data with trend, weekly "
    "seasonality, and noise. LSTM achieved **MAE=17.9 (MAPE 4.3%)** vs "
    "ARIMA MAE=62.6 (MAPE 14.9%) on the 60-day hold-out test set.",
    icon="📈",
)


@st.cache_resource
def get_model():
    try:
        return load_model_and_scale()
    except Exception as e:
        return e


@st.cache_data
def get_data():
    df = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "raw", "daily_sales.csv"),
        parse_dates=["date"],
    )
    return df


model_result = get_model()
if isinstance(model_result, Exception):
    st.error("### Model Loading Error")
    st.error(str(model_result))
    st.info("To train a new model, run the following commands in your terminal:")
    st.code("python src/data_preprocessing.py && python src/train_model.py")
    st.stop()
model, mn, mx = model_result

df = get_data()

st.divider()
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Controls")
    n_days     = st.slider("Days to forecast", 7, 30, 14)
    history_days = st.slider("Days of history to show", 30, 180, 90)

with col1:
    recent_sales = df["sales"].values[-30:]
    forecast     = forecast_next_n_days(recent_sales, n_days=n_days,
                                         model=model, mn=mn, mx=mx)

    last_date       = df["date"].iloc[-1]
    forecast_dates  = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_days)
    history_slice   = df.tail(history_days)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history_slice["date"], history_slice["sales"],
            label="Historical Sales", color="black", linewidth=1.2)
    ax.plot(forecast_dates, forecast,
            label=f"LSTM Forecast ({n_days} days)", color="#1f77b4",
            linewidth=2, linestyle="--", marker="o", markersize=3)
    ax.axvline(last_date, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("Sales History + LSTM Forecast")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# Forecast table
st.subheader(f"Forecast — Next {n_days} Days")
forecast_df = pd.DataFrame({
    "Date": forecast_dates.strftime("%Y-%m-%d"),
    "Forecast Sales": [f"{v:.1f}" for v in forecast],
})
st.dataframe(forecast_df, hide_index=True)

# Model comparison chart
st.subheader("Model Comparison (Test Set)")
img_path = os.path.join(PROJECT_ROOT, "images", "forecast_comparison.png")
if os.path.exists(img_path):
    # pyrefly: ignore [unexpected-keyword]
    st.image(img_path, use_container_width=True)
