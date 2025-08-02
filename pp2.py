import streamlit as st # type: ignore
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from prophet import Prophet
import numpy as np
from streamlit_lottie import st_lottie
import requests
import pydeck as pdk
import plotly.express as px
import os
import base64
# from io import BytesIO
# import imageio
st.set_page_config(page_title="Lemon Mandi Dashboard", layout="wide")
st.title("Lemon Mandi Dashboard")

# Load dataset
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file was not found at {file_path}. Please make sure the file exists.")
        return pd.DataFrame()  # Return empty df to prevent app crash

# Remove outliers
@st.cache_data
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Get full path to your cleaned data
DATA_FILE_PATH = r"C:\Users\ASUS\Downloads\LemonPrices.csv"
df = load_data(DATA_FILE_PATH)

@st
def generate_forecast(_df):
    """Generates and returns a Prophet forecast."""
    # Prepare data for Prophet
    forecast_df = _df.groupby('Date')['Modal Price(Rs./Quintal)'].mean().reset_index()
    forecast_df = forecast_df.rename(columns={'Date': 'ds', 'Modal Price(Rs./Quintal)': 'y'})

    # Initialize and fit the model
    model = Prophet()
    model.fit(forecast_df)

    # Create future dataframe for 2 quarters (approx. 182 days)
    future = model.make_future_dataframe(periods=182)
    forecast = model.predict(future)
    return model, forecast

if df is not None:
    df_cleaned = df.dropna()
    df_cleaned = remove_outliers_iqr(df_cleaned, 'Modal Price(Rs./Quintal)')
    df_cleaned = remove_outliers_iqr(df_cleaned, 'Arrivals (Tonnes)')

    # Metric Cards
    st.subheader("Quick Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df)}")
    col2.metric("States Covered", f"{df['State'].nunique()}")
    col3.metric("Mandis Tracked", f"{df['Market'].nunique()}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Preview", "Trends", "Data Quality", "Geo Insights", "Forecast"])

    with tab1:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

    with tab2:
        st.subheader("Price & Arrival Trends")
        selected_states = st.multiselect("Choose States", df_cleaned['State'].unique(), default=[df_cleaned['State'].unique()[0]])
        filtered_df = df_cleaned[df_cleaned['State'].isin(selected_states)]

        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        sns.lineplot(data=filtered_df, x='Date', y='Modal Price(Rs./Quintal)', hue='State', ax=ax[0])
        ax[0].set_title('Modal Price Over Time')

        sns.lineplot(data=filtered_df, x='Date', y='Arrivals (Tonnes)', hue='State', ax=ax[1])
        ax[1].set_title('Arrivals Over Time')
        st.pyplot(fig)

        st.subheader("Monthly Avg Price (All India)")
        monthly_avg = df_cleaned.groupby(['Year', 'Month'])['Modal Price(Rs./Quintal)'].mean().reset_index()
        monthly_avg['Date'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(DAY=1))

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=monthly_avg, x='Date', y='Modal Price(Rs./Quintal)', marker='o', ax=ax2)
        st.pyplot(fig2)

    with tab3:
        st.subheader("Data Quality Report")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Missing Values Count:**")
            st.write(df.isnull().sum())
        with col2:
            st.markdown("**Descriptive Stats:**")

    with tab4:
        st.subheader(" Interactive Geo Price Heatmap")
        st.markdown("Average lemon prices by State and Month")
        heatmap_df = df_cleaned.groupby(['State', 'Month'])['Modal Price(Rs./Quintal)'].mean().reset_index()
        heatmap_pivot = heatmap_df.pivot(index='State', columns='Month', values='Modal Price(Rs./Quintal)')

        fig3, ax3 = plt.subplots(figsize=(10, 8))
        sns.heatmap(heatmap_pivot, cmap="YlOrBr", annot=True, fmt=".0f", linewidths=.5)
        st.pyplot(fig3)

        

    with tab5:
        st.subheader(" Forecasting Modal Price (Next 2 Quarters)")
        # Generate forecast with a spinner and caching
        with st.spinner(" Casting forecasting spells... This may take a moment."):
            model, forecast = generate_forecast(df_cleaned)

        st.success("Forecast complete.")

        # Display forecast data
        st.subheader("Forecasted Prices for the Next 6 Months")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(182))

        # Display interactive forecast plot
        st.subheader("Interactive Forecast Plot")
        from prophet.plot import plot_plotly
        fig_plotly = plot_plotly(model, forecast)
        fig_plotly.update_layout(
            title="Price Forecast for the Next 6 Months",
            xaxis_title="Date",
            yaxis_title="Modal Price (Rs./Quintal)"
        )
        st.plotly_chart(fig_plotly, use_container_width=True)

        # Display forecast components
        st.subheader("Forecast Components")
        from prophet.plot import plot_components_plotly
        fig_components = plot_components_plotly(model, forecast)
        st.plotly_chart(fig_components, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.download_button(" Download Cleaned Data", data=df_cleaned.to_csv(index=False), file_name="cleaned_lemon_data.csv", mime='text/csv')
else:
    st.warning("Upload a CSV file in sidebar to get started ")
