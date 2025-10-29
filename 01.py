import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("🥔 Punjab Potato Spot Price Dashboard (2020–2025)")

# --- Load and clean data ---
df = pd.read_csv(r"c:\Users\abdul\Downloads\amCharts (2).csv")
df = df.drop(['spot_price_2026', 'spot_price_2027'], axis=1)
df["Date"] = pd.to_datetime(df["Date"])

# --- Filter actual 2025 data ---
df_2025_actual = df[df["spot_price_2025"] > 0].copy().reset_index(drop=True)
x_train = df_2025_actual.index.values.reshape(-1, 1)
y_train = df_2025_actual["spot_price_2025"]

# --- Linear Regression model (optional) ---
model = LinearRegression()
model.fit(x_train, y_train)

# --- Weekly seasonality calculation ---
seasonal_prices = []
for year in ["2020", "2021", "2022", "2023", "2024"]:
    prices = df[f"spot_price_{year}"].values[:52]
    for week, price in enumerate(prices, 1):
        seasonal_prices.append([week, price])
season_df = pd.DataFrame(seasonal_prices, columns=["Week", "Price"])
avg_season = season_df.groupby("Week")["Price"].mean()

# --- Combine all data for plotting ---
all_data = []

# Historical data (2020–2024)
for year in ["2020", "2021", "2022", "2023", "2024"]:
    prices = df[f"spot_price_{year}"].values[:52]
    for week, price in enumerate(prices, 1):
        all_data.append([week, price, year, "Actual"])

# Actual 2025 (weeks 1–29)
actual_2025 = df["spot_price_2025"][df["spot_price_2025"] > 0].values
for week, price in enumerate(actual_2025, 1):
    all_data.append([week, price, "2025", "Actual"])

# Predicted 2025 (weeks 30–52 using seasonality)
for week in range(30, 53):
    predicted_price = avg_season[week]
    all_data.append([week, predicted_price, "2025", "Predicted"])

plot_data = pd.DataFrame(all_data, columns=["Week", "Price", "Year", "Type"])

# --- Optional: show prediction table ---
week_labels = [f"2025-{wk}" for wk in range(30, 53)]
future_dates = pd.date_range("2025-07-21", periods=23, freq="W-MON")
predicted_prices = [round(avg_season[wk], 1) for wk in range(30, 53)]

future_df = pd.DataFrame({
    "Week": week_labels,
    "Date": future_dates.strftime("%Y-%m-%d"),
    "Predicted Price (2025)": predicted_prices
})

st.subheader("📋 2025 Predicted Spot Prices (Week 30–52)")
st.dataframe(future_df, use_container_width=True)

# --- Plotting ---
st.subheader("📈 Spot Prices (2020–2025) with 2025 Prediction Highlighted")

# 2020–2024 actuals
actual_data = plot_data[plot_data["Type"] == "Actual"]
predicted_data = plot_data[plot_data["Type"] == "Predicted"]

fig = px.line(
    actual_data[actual_data["Year"] != "2025"],
    x="Week",
    y="Price",
    color="Year",
    title="Potato Spot Prices with Highlighted 2025 Forecast",
    labels={"Price": "Spot Price", "Week": "Week of Year"},
    template="plotly_white",
)

# Add 2025 (actual + predicted)
combined_2025 = plot_data[plot_data["Year"] == "2025"]
fig.add_scatter(
    x=combined_2025["Week"],
    y=combined_2025["Price"],
    mode="lines+markers",
    name="2025 (Actual + Pr edicted)",
    line=dict(color="red", width=3),
    marker=dict(size=5),
    showlegend=True
)

# Highlight prediction range
fig.add_vrect(
    x0=29.5, x1=52,
    fillcolor="lightgray",
    opacity=0.25,
    annotation_text="Predicted Region (2025)",
    annotation_position="top left",
    line_width=0
)

fig.update_layout(
    hovermode="x unified",
    legend_title_text="Year",
    xaxis=dict(tickmode='linear', tick0=1, dtick=4, range=[1, 52]),
    yaxis=dict(title="Price", range=[0, plot_data["Price"].max() * 1.1]),
    height=600
)

# Show chart in Streamlit
st.plotly_chart(fig, use_container_width=True)
