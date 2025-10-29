import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Punjab Commodity Dashboard", layout="wide")

st.title("Commodity Spot Price Dashboard")

# Set consistent color scheme
BG_COLOR = "#f0f2f6"
PLOT_BGCOLOR = "#FFFFFF"
FONT_COLOR = "#000000"

# -----------------------
# Load Potato Data
# -----------------------
df = pd.read_csv(r"c:\Users\abdul\Downloads\amCharts (2).csv")
df = df.drop(['spot_price_2026', 'spot_price_2027'], axis=1)
df["Date"] = pd.to_datetime(df["Date"])

# -----------------------
# Load Other Commodities
# -----------------------
df2 = pd.read_csv(r"c:\Users\abdul\Downloads\amchart,432.csv")
df2 = df2.drop(['spot_price_2026', 'spot_price_2027'], axis=1)
df2["Date"] = pd.to_datetime(df2["Date"])

commodity_list = [
    "Potato", 'Flour', 'Milk',
    "Baspati Ghee-RBD", "Baspati Ghee-Soft Oil 10%", "Baspati Ghee-Soft Oil 20%",
    "Chicken_Broiler", "Daal Chana-Bareek", "Daal Chana-Moti",'DAP','Chaki','basin','Lemon',
    "Onion", "Potato-New", "Potato-Old","Sugar", "Tomato",'Moong',"Naan","Masoor"
]

# -----------------------
# Define flood weeks (no multipliers)
flood_weeks = [34, 35, 36, 37, 38, 39, 40, 41]
post_flood_weeks = [42,43,44,45, 46]  # Weeks after flood with residual impact

# -----------------------
# Commodity Dropdown
# -----------------------
selected_com = st.selectbox("Select a commodity:", commodity_list)

# Custom template
custom_template = {
    "layout": {
        "paper_bgcolor": PLOT_BGCOLOR,
        "plot_bgcolor": PLOT_BGCOLOR,
        "font": {"color": FONT_COLOR, "size": 12},
        "xaxis": {
            "gridcolor": "#e6e6e6",
            "linecolor": "#e6e6e6",
            "zerolinecolor": "#e6e6e6",
            "title_font": {"size": 14, "color": "black"},
            "tickfont": {"color": "black"},
            "title_standoff": 15
        },
        "yaxis": {
            "gridcolor": "#e6e6e6",
            "linecolor": "#e6e6e6",
            "zerolinecolor": "#e6e6e6",
            "title_font": {"size": 14, "color": "black"},
            "tickfont": {"color": "black"},
            "title_standoff": 15
        }
    }
}

# -----------------------
# Flood Impact Summary
# -----------------------
st.subheader("Flood Impact Analysis")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Flood-affected Weeks", len(flood_weeks))
with col2:
    st.metric("Flood Period", f"Weeks {min(flood_weeks)} to {max(flood_weeks)}")
with col3:
    st.metric("Post-Flood Impact Weeks", len(post_flood_weeks))

# Create a visual showing flood weeks and post-flood weeks
flood_df = pd.DataFrame({
    "Week": flood_weeks + post_flood_weeks,
    "Period": ["Flood"] * len(flood_weeks) + ["Post-Flood"] * len(post_flood_weeks)
})
impact_fig = px.scatter(flood_df, x="Week", y="Period", 
                    title="Flood and Post-Flood Affected Weeks",
                    color="Period",
                    color_discrete_sequence=["red", "orange"])
impact_fig.update_layout(template=custom_template, showlegend=True)
st.plotly_chart(impact_fig, use_container_width=True)

# -----------------------
# Function to calculate flood impact
# -----------------------
def calculate_flood_impact(commodity_name, flood_weeks, post_flood_weeks):
    """
    Calculate the average price increase during flood weeks and post-flood weeks
    for a commodity based on historical data
    """
    if commodity_name == "Potato":
        data_source = df
    else:
        data_source = df2[df2["commodity_name"] == commodity_name]
    
    # Get historical years
    years = ["2020", "2021", "2022", "2023", "2024"]
    flood_impacts = []
    post_flood_impacts = []
    
    for year in years:
        year_col = f"spot_price_{year}"
        if year_col in data_source.columns:
            # Get average price for normal weeks (non-flood, non-post-flood)
            normal_weeks = [w for w in range(1, 53) if w not in flood_weeks and w not in post_flood_weeks]
            normal_prices = [data_source[year_col].iloc[w-1] for w in normal_weeks if w-1 < len(data_source)]
            avg_normal = np.nanmean(normal_prices) if normal_prices else 0
            
            # Get prices for flood weeks
            flood_prices = []
            for week in flood_weeks:
                if week-1 < len(data_source):
                    flood_prices.append(data_source[year_col].iloc[week-1])
            
            # Get prices for post-flood weeks
            post_flood_prices = []
            for week in post_flood_weeks:
                if week-1 < len(data_source):
                    post_flood_prices.append(data_source[year_col].iloc[week-1])
            
            # Calculate impacts if we have data
            if flood_prices and avg_normal > 0:
                avg_flood = np.nanmean(flood_prices)
                flood_impact = (avg_flood - avg_normal) / avg_normal
                flood_impacts.append(flood_impact)
            
            if post_flood_prices and avg_normal > 0:
                avg_post_flood = np.nanmean(post_flood_prices)
                post_flood_impact = (avg_post_flood - avg_normal) / avg_normal
                post_flood_impacts.append(post_flood_impact)
    
    # Return average impacts across all years
    flood_impact_factor = np.nanmean(flood_impacts) if flood_impacts else 0.2
    post_flood_impact_factor = np.nanmean(post_flood_impacts) if post_flood_impacts else flood_impact_factor * 0.5
    
    return flood_impact_factor, post_flood_impact_factor

# -----------------------
# Case 1: Potato
# -----------------------
if selected_com == "Potato":
    # Calculate flood impact for potato
    flood_impact_factor, post_flood_impact_factor = calculate_flood_impact("Potato", flood_weeks, post_flood_weeks)
    
    df_2025_actual = df[df["spot_price_2025"] > 0].copy().reset_index(drop=True)

    seasonal_prices = []
    for year in ["2020", "2021", "2022", "2023", "2024"]:
        prices = df[f"spot_price_{year}"].values[:52]
        for week, price in enumerate(prices, 1):
            seasonal_prices.append([week, price])
    season_df = pd.DataFrame(seasonal_prices, columns=["Week", "Price"])
    avg_season = season_df.groupby("Week")["Price"].mean()

    all_data = []
    for year in ["2020", "2021", "2022", "2023", "2024"]:
        prices = df[f"spot_price_{year}"].values[:52]
        for week, price in enumerate(prices, 1):
            all_data.append([week, price, year, "Actual"])

    # Actual 2025
    actual_2025 = df["spot_price_2025"][df["spot_price_2025"] > 0].values
    for week, price in enumerate(actual_2025, 1):
        all_data.append([week, price, "2025", "Actual"])

    # Predicted 2025 (week 30 onward with flood adjustment)
    for week in range(30, 53):
        predicted_price = avg_season[week]
        if week in flood_weeks:
            predicted_price *= (1 + flood_impact_factor)  # Apply flood impact
        elif week in post_flood_weeks:
            predicted_price *= (1 + post_flood_impact_factor)  # Apply post-flood impact
        all_data.append([week, predicted_price, "2025", "Predicted"])

    plot_data = pd.DataFrame(all_data, columns=["Week", "Price", "Year", "Type"])

    # Base years
    fig_full = px.line(
        plot_data[(plot_data["Type"] == "Actual") & (plot_data["Year"] != "2025")],
        x="Week", y="Price", color="Year",
        title=f"Trend of spot price – Potato (2020–2025) - Flood Impact: {flood_impact_factor*100:.1f}%, Post-Flood: {post_flood_impact_factor*100:.1f}%",
        template=custom_template,
        labels={"Week": "Weeks", "Price": "Price"},
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    # --- Actual 2025 solid ---
    actual_25 = plot_data[(plot_data["Year"] == "2025") & (plot_data["Type"] == "Actual")]
    fig_full.add_trace(go.Scatter(
        x=actual_25["Week"],
        y=actual_25["Price"],
        mode="lines",
        name="2025 Actual",
        line=dict(color="black", width=3),
        showlegend=True
    ))

    # --- Predicted 2025 dashed (connected) ---
    pred_25 = plot_data[(plot_data["Year"] == "2025") & (plot_data["Type"] == "Predicted")]
    if not actual_25.empty and not pred_25.empty:
        fig_full.add_trace(go.Scatter(
            x=[actual_25["Week"].iloc[-1]] + pred_25["Week"].tolist(),
            y=[actual_25["Price"].iloc[-1]] + pred_25["Price"].tolist(),
            mode="lines",
            name="2025 Prediction (Flood Adjusted)",
            line=dict(color="red", width=3, dash="dash"),
            showlegend=True
        ))

    # Highlight prediction area
    fig_full.add_vrect(
        x0=30, x1=52,
        fillcolor="lightgray", opacity=0.3,
        layer="below", line_width=0
    )

    # Flood shading for each flood week
    for week in flood_weeks:
        fig_full.add_vrect(
            x0=week-0.5, x1=week+0.5,
            fillcolor="red", opacity=0.2,
            layer="below", line_width=0,
            annotation_text=f"+{flood_impact_factor*100:.0f}%",
            annotation_position="top"
        )
    
    # Post-flood shading for each post-flood week
    for week in post_flood_weeks:
        fig_full.add_vrect(
            x0=week-0.5, x1=week+0.5,
            fillcolor="orange", opacity=0.2,
            layer="below", line_width=0,
            annotation_text=f"+{post_flood_impact_factor*100:.0f}%",
            annotation_position="top"
        )

    # Layout
    fig_full.update_layout(
        xaxis=dict(
            title="Weeks", range=[1, 52],
            tickmode="linear", dtick=1,
            title_font=dict(size=14, color="black",family="Arial",),
            tickfont=dict(size=12, color="black",family="Arial",)
        ),
        yaxis=dict(
            title="Price in Rupees",
            title_font=dict(size=14, color="black",family="Arial",),
            tickfont=dict(size=12, color="black",family="Arial",)
        ),
        title_font=dict(size=18, color="black",family="Arial",),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.25,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black", borderwidth=1,
            font=dict(size=11, color="black", family="Arial Bold")
        )
    )
    st.plotly_chart(fig_full, use_container_width=True)

    # Show impact details
    st.subheader(f"Flood Impact on {selected_com} Prices")
    
    impact_details = []
    for week in range(30, 53):
        if week in flood_weeks:
            base_price = avg_season[week]
            impacted_price = base_price * (1 + flood_impact_factor)
            impact_type = "Flood"
            impact_percent = flood_impact_factor * 100
        elif week in post_flood_weeks:
            base_price = avg_season[week]
            impacted_price = base_price * (1 + post_flood_impact_factor)
            impact_type = "Post-Flood"
            impact_percent = post_flood_impact_factor * 100
        else:
            continue
            
        price_diff = impacted_price - base_price
        impact_details.append({
            "Week": week,
            "Impact Type": impact_type,
            "Base Price (Rs)": round(base_price, 2),
            "Impacted Price (Rs)": round(impacted_price, 2),
            "Price Increase (Rs)": round(price_diff, 2),
            "Impact (%)": f"{impact_percent:.1f}%"
        })
    
    if impact_details:
        impact_df = pd.DataFrame(impact_details)
        st.dataframe(impact_df, use_container_width=True)

# -----------------------
# Case 2: Other Commodities
# -----------------------
else:
    # Calculate flood impact for this commodity
    flood_impact_factor, post_flood_impact_factor = calculate_flood_impact(selected_com, flood_weeks, post_flood_weeks)
    
    cdf = df2[df2["commodity_name"] == selected_com].reset_index(drop=True)
    df_2025 = cdf[cdf["spot_price_2025"] > 0].reset_index(drop=True)
    num_weeks = len(df_2025)

    if num_weeks >= 2:
        # Train linear regression
        x_train = np.arange(num_weeks).reshape(-1, 1)
        y_train = df_2025["spot_price_2025"]
        model = LinearRegression().fit(x_train, y_train)

        # Predict trend
        x_pred = np.arange(num_weeks, num_weeks + 20).reshape(-1, 1)
        trend_pred = model.predict(x_pred)

        # Seasonal averages
        seasonal_prices = []
        for yr in ["2020","2021","2022","2023","2024"]:
            if f"spot_price_{yr}" in cdf:
                prices = cdf[f"spot_price_{yr}"].values[:52]
                if len(prices) < 52:
                    prices = np.append(prices, [np.nan]*(52-len(prices)))
                seasonal_prices.extend(prices)
        seasonal_matrix = np.array(seasonal_prices).reshape(5, 52)
        avg_season = np.nanmean(seasonal_matrix, axis=0)

        # Blend trend + seasonal prediction (with flood adjustment)
        predicted_prices = []
        for i in range(20):
            wk = num_weeks + i
            season_wk = wk % 52
            combined = (trend_pred[i] + avg_season[season_wk]) / 2
            if wk in flood_weeks:
                combined *= (1 + flood_impact_factor)  # Apply flood impact
            elif wk in post_flood_weeks:
                combined *= (1 + post_flood_impact_factor)  # Apply post-flood impact
            predicted_prices.append(round(combined, 2))

        # --- Build Figure ---
        fig = go.Figure()
        colors = ["blue", "red", "green", "purple", "orange"]
        for yr, color in zip(["2020","2021","2022","2023","2024"], colors):
            if f"spot_price_{yr}" in cdf:
                vals = cdf[f"spot_price_{yr}"].values[:52]
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(vals)+1)), y=vals,
                    mode="lines", name=yr,
                    line=dict(color=color), connectgaps=True
                ))

        # --- Actual 2025 solid ---
        fig.add_trace(go.Scatter(
            x=list(range(1, num_weeks+1)),
            y=df_2025["spot_price_2025"],
            mode="lines",
            name="2025 Actual",
            line=dict(color="black", width=3),
            showlegend=True
        ))

        # --- Predicted 2025 dashed (connected, flood adjusted) ---
        fig.add_trace(go.Scatter(
            x=[num_weeks] + list(range(num_weeks+1, num_weeks+len(predicted_prices)+1)),
            y=[df_2025["spot_price_2025"].iloc[-1]] + predicted_prices,
            mode="lines",
            name="2025 Prediction (Flood Adjusted)",
            line=dict(color="red", width=3, dash="dash"),
            showlegend=True
        ))

        # Highlight prediction area
        fig.add_vrect(
            x0=num_weeks, x1=num_weeks + len(predicted_prices),
            fillcolor="lightgray", opacity=0.3,
            layer="below", line_width=0
        )

        # Flood shading for each flood week in prediction range
        for week in flood_weeks:
            if week >= num_weeks and week <= (num_weeks + len(predicted_prices)):
                fig.add_vrect(
                    x0=week-0.5, x1=week+0.5,
                    fillcolor="red", opacity=0.2,
                    layer="below", line_width=0,
                    annotation_text=f"+{flood_impact_factor*100:.0f}%",
                    annotation_position="top"
                )
        
        # Post-flood shading for each post-flood week in prediction range
        for week in post_flood_weeks:
            if week >= num_weeks and week <= (num_weeks + len(predicted_prices)):
                fig.add_vrect(
                    x0=week-0.5, x1=week+0.5,
                    fillcolor="orange", opacity=0.2,
                    layer="below", line_width=0,
                    annotation_text=f"+{post_flood_impact_factor*100:.0f}%",
                    annotation_position="top"
                )

        # Layout
        fig.update_layout(
            title=f"{selected_com} – Spot Price Trend (2020–2025) - Flood Impact: {flood_impact_factor*100:.1f}%, Post-Flood: {post_flood_impact_factor*100:.1f}%",
            xaxis=dict(
                title="Weeks", range=[1, 52],
                tickmode="linear", dtick=1,
                title_font=dict(size=14, color="black", family="Arial"),
                tickfont=dict(size=12, color="black", family="Arial")
            ),
            yaxis=dict(
                title="Price in Rupees",
                title_font=dict(size=14, color="black", family="Arial"),
                tickfont=dict(size=12, color="black", family="Arial")
            ),
            title_font=dict(size=18, color="black", family="Arial"),
            template=custom_template,
            paper_bgcolor=PLOT_BGCOLOR,
            plot_bgcolor=PLOT_BGCOLOR,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.25,
                xanchor="center", x=0.5,
                font=dict(size=14, color="black", family="Arial Bold")
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show flood impact details for this commodity
        st.subheader(f"Flood Impact on {selected_com} Prices")
        
        # Calculate and display impact details
        impact_details = []
        for i in range(len(predicted_prices)):
            wk = num_weeks + i
            if wk in flood_weeks or wk in post_flood_weeks:
                base_price = (trend_pred[i] + avg_season[wk % 52]) / 2
                
                if wk in flood_weeks:
                    impacted_price = base_price * (1 + flood_impact_factor)
                    impact_type = "Flood"
                    impact_percent = flood_impact_factor * 100
                else:  # post_flood_weeks
                    impacted_price = base_price * (1 + post_flood_impact_factor)
                    impact_type = "Post-Flood"
                    impact_percent = post_flood_impact_factor * 100
                
                price_diff = impacted_price - base_price
                impact_details.append({
                    "Week": wk,
                    "Impact Type": impact_type,
                    "Base Price (Rs)": round(base_price, 2),
                    "Impacted Price (Rs)": round(impacted_price, 2),
                    "Price Increase (Rs)": round(price_diff, 2),
                    "Impact (%)": f"{impact_percent:.1f}%"
                })
        
        if impact_details:
            impact_df = pd.DataFrame(impact_details)
            st.dataframe(impact_df, use_container_width=True)
        else:
            st.info("No flood impact in the prediction range for this commodity.")
            
    else:
        st.warning("Not enough 2025 data for prediction.") 