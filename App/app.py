# =====================================================
# 🎮 GameVision - Smart Video Game Sales Dashboard
# =====================================================

import streamlit as st
import pandas as pd
import joblib
from prophet import Prophet
from plots import plot_sales_trend, plot_forecast
from utils import load_data

# --------------------------
# 🎛️ Page Configuration
# --------------------------
st.set_page_config(page_title="GameVision Dashboard", layout="wide")

# --------------------------
# 🧠 Load Data & Model
# --------------------------
df = load_data('../Data/vgsales_clean.csv')
model = joblib.load('../Models/prophet_forecast.pkl')

st.title("🎮 GameVision – Smart Video Game Sales Intelligence Dashboard")
st.markdown("Analyze, Forecast, and Visualize Global Game Sales Trends.")

# --------------------------
# 🧩 Sidebar Filters
# --------------------------
st.sidebar.header("🔎 Filter Options")
genre = st.sidebar.selectbox("🎮 Select Genre", sorted(df['Genre'].unique()))
platform = st.sidebar.selectbox("💻 Select Platform", sorted(df['Platform'].unique()))

filtered_df = df[(df['Genre'] == genre) & (df['Platform'] == platform)]

if filtered_df.empty:
    st.warning("No data available for this combination. Try another.")
else:
    trend_data = filtered_df.groupby('Year_of_Release')['Global_Sales'].sum().reset_index()
    st.subheader(f"📊 Sales Trend for {genre} on {platform}")
    st.plotly_chart(plot_sales_trend(trend_data), use_container_width=True)

# --------------------------
# 🔮 Forecast Visualization
# --------------------------
st.subheader("🔮 Global Sales Forecast (Prophet)")

# Predict for next 5 years
future = model.make_future_dataframe(periods=5, freq='Y')
forecast = model.predict(future)

st.plotly_chart(plot_forecast(forecast), use_container_width=True)

# --------------------------
# 📘 Dataset Reference
# --------------------------
st.markdown("---")
st.markdown("📘 **Dataset:** [Kaggle Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales)")
st.markdown("© 2025 GameVision | Built with ❤️ using Streamlit, Prophet, and Python.")
