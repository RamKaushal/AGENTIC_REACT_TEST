import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta


actual_df = pd.read_excel(r"/content/Call_Volume_Data_2020_to_2025.xlsx")
actual_df["DAY_OF_WEEK"] = actual_df["REPORT_DT"].dt.day_name()
actual_df["Week_Start"] = actual_df["REPORT_DT"] - pd.to_timedelta(actual_df["REPORT_DT"].dt.weekday, unit='d')


forecast_df = pd.read_excel(r"/content/AGENTIC_LSTM_FORECAST.xlsx")
forecast_df["DAY_OF_WEEK"] = forecast_df["Date"].dt.day_name()


aligned_actual = actual_df.tail(25).copy()
aligned_actual = aligned_actual.reset_index(drop=True)
forecast_df["Actual"] = aligned_actual["TOTAL_OFFERED_CALL_VOLUME"]

comparison = forecast_df.copy()
comparison["Abs Error"] = abs(comparison["Forecast"] - comparison["Actual"])
comparison["APE"] = 100 * comparison["Abs Error"] / comparison["Actual"]
daily_mape = round(comparison["APE"].mean(), 2)
mae = round(comparison["Abs Error"].mean(), 2)
rmse = round(np.sqrt(np.mean((comparison["Forecast"] - comparison["Actual"])**2)), 2)

weekly_summary = (
    actual_df.groupby("Week_Start")
    .agg(Volume=("TOTAL_OFFERED_CALL_VOLUME", "sum"))
    .reset_index()
    .sort_values("Week_Start", ascending=False)
    .head(5)
    .sort_values("Week_Start")
)
weekly_summary["AHT"] = np.random.randint(150, 451, size=5)
weekly_summary["NORM (%)"] = np.round(
    (weekly_summary["Forecast"] - weekly_summary["Volume"]) / weekly_summary["Volume"] * 100, 2
)


st.set_page_config(page_title="📞 Call Volume Forecast Dashboard")
st.title("📞 Forecast vs Actual Comparison - July Data")


st.markdown("### 📊 Forecast Accuracy Metrics (Recent 25 Days)")
col1, col2, col3 = st.columns(3)
col1.metric("📉 MAE", f"{mae}")
col2.metric("📈 RMSE", f"{rmse}")
col3.metric("⚠️ Daily MAPE", f"{daily_mape}%")


st.markdown("### 📅 Week-over-Week Summary (Last 5 Weeks)")
st.dataframe(weekly_summary.rename(columns={"Week_Start": "Week Start Date"}), use_container_width=True)


comparison_plot_df = comparison[["Date", "Forecast", "Actual"]].melt(id_vars="Date", var_name="Type", value_name="Volume")
fig = px.line(
    comparison_plot_df,
    x="Date",
    y="Volume",
    color="Type",
    title="📈 Daily Volume vs Forecast (Most Recent 25 Days)",
    labels={"Value": "Call Volume", "Date": "Date"},
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Call Volume",
    template="plotly_white",
    xaxis=dict(rangeslider=dict(visible=True), type="date")
)
st.plotly_chart(fig, use_container_width=True)
