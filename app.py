import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns  # type: ignore
import streamlit as st   # type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAX # type: ignore

st.set_page_config(layout="wide", page_title="Lemon Price Analysis")
sns.set(style="whitegrid")

st.title("Lemon Price Analysis and Forecasting")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload LemonPrices.csv", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Arrivals (Tonnes)', 'Date'], how='all')
    df = df.sort_values(['Market','Date'])

    # Remove anomalies: >200% change
    def anomaly(series):
        return series.pct_change().abs() > 2
    df['pa'] = df.groupby('Market')['Modal Price(Rs./Quintal)'].transform(anomaly)
    df['qa'] = df.groupby('Market')['Arrivals (Tonnes)'].transform(anomaly)
    df = df[~(df['pa'] | df['qa'])]

    st.success("Data loaded and cleaned!")

    # --- Seasonal trends (All commodities) ---
    st.header("Seasonal Trends")
    df['Month'] = df['Date'].dt.month
    monthly = df.groupby('Month').agg(
        avg_price=('Modal Price(Rs./Quintal)','mean'),
        total_arrivals=('Arrivals (Tonnes)','sum')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(8,4))
    sns.lineplot(x='Month', y='avg_price', data=monthly, marker='o', color='blue', ax=ax1)
    ax1.set_ylabel("Avg Price")
    ax2 = ax1.twinx()
    sns.barplot(x='Month', y='total_arrivals', data=monthly, alpha=0.3, color='orange', ax=ax2)
    ax2.set_ylabel("Total Arrivals")
    st.pyplot(fig)

    # --- Lemon Trends ---
    st.header("Lemon Price Trends (All India & Top 5 States)")
    df['M_Period'] = df['Date'].dt.to_period('M')
    monthly_lemon = df.groupby('M_Period')['Modal Price(Rs./Quintal)'].mean().dropna()

    # Plot All India
    fig, ax = plt.subplots(figsize=(10,4))
    monthly_lemon.plot(marker='o', ax=ax)
    ax.set_title("Monthly Avg Lemon Price - All India")
    st.pyplot(fig)

    # Top 5 states
    top5 = df['State'].value_counts().head(5).index
    pivot = df[df['State'].isin(top5)].groupby(
        ['M_Period','State']
    )['Modal Price(Rs./Quintal)'].mean().unstack()
    fig, ax = plt.subplots(figsize=(10,4))
    pivot.plot(marker='o', ax=ax)
    ax.set_title("Monthly Avg Lemon Price - Top 5 States")
    st.pyplot(fig)

    # --- Forecast (All India) ---
    st.header("6-Month Price Forecast (All India)")
    ts = monthly_lemon.sort_index().to_timestamp()
    series = ts.dropna()
    model = SARIMAX(series, order=(1,1,1))
    fit = model.fit(disp=False)

    forecast_res = fit.get_forecast(steps=6)
    forecast = forecast_res.predicted_mean
    conf = forecast_res.conf_int()
    future_dates = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1),
                                  periods=6, freq='MS')

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(series.index, series.values, label='Price')
    ax.plot(future_dates, forecast, label='Forecast', color='red')
    ax.fill_between(future_dates, conf.iloc[:,0], conf.iloc[:,1], color='pink', alpha=0.3)
    ax.set_title("6-Month Lemon Price Forecast (All India)")
    ax.legend()
    st.pyplot(fig)

    # Download cleaned data
    st.header("Download Cleaned Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "Cleaned_LemonPrices.csv", "text/csv")

else:
    st.info("Upload a CSV file to get started.")
