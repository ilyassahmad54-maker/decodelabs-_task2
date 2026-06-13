import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Climate Intelligence Dashboard", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("climate_change.csv")
    return df

df = load_data()

# -----------------------------
# TITLE
# -----------------------------
st.title("🌍 Climate Intelligence Dashboard ")
st.markdown("AI-powered analysis of global climate trends")

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

year_range = st.sidebar.slider(
    "Select Year Range",
    int(df["year"].min()),
    int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max()))
)

country = st.sidebar.selectbox("Select Country", ["All"] + list(df["country"].unique()))

filtered = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

if country != "All":
    filtered = filtered[filtered["country"] == country]

# -----------------------------
# KPI SECTION
# -----------------------------
st.subheader("📊 Key Climate Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Temperature", round(filtered["global_avg_temperature"].mean(), 2))
col2.metric("CO₂ (ppm)", round(filtered["co2_concentration_ppm"].mean(), 2))
col3.metric("Climate Risk", round(filtered["climate_risk_index"].mean(), 2))
col4.metric("Heatwave Days", int(filtered["heatwave_days"].mean()))

st.markdown("---")

# -----------------------------
# 🌡️ TEMPERATURE TREND
# -----------------------------
st.subheader("🌡️ Global Temperature Trend")

temp = filtered.groupby("year")["global_avg_temperature"].mean().reset_index()

fig = px.line(temp, x="year", y="global_avg_temperature",
              title="Temperature Over Time")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 🏭 CO2 TREND
# -----------------------------
st.subheader("🏭 CO₂ Concentration Trend")

co2 = filtered.groupby("year")["co2_concentration_ppm"].mean().reset_index()

fig = px.line(co2, x="year", y="co2_concentration_ppm",
              color_discrete_sequence=["red"],
              title="CO₂ Levels Over Time")

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 🌍 COUNTRY RISK MAP (IMPORTANT)
# -----------------------------
st.subheader("🌍 Climate Risk by Country")

map_df = df.groupby("country")["climate_risk_index"].mean().reset_index()

fig = px.choropleth(
    map_df,
    locations="country",
    locationmode="country names",
    color="climate_risk_index",
    color_continuous_scale="Reds",
    title="Global Climate Risk Map"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# ⚡ ENERGY COMPARISON
# -----------------------------
st.subheader("⚡ Renewable vs Fossil Fuel")

fig = px.scatter(
    filtered,
    x="renewable_energy_share",
    y="fossil_fuel_consumption",
    color="climate_risk_index",
    size="co2_concentration_ppm",
    title="Energy Transition Analysis"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 🔥 CORRELATION HEATMAP
# -----------------------------
st.subheader("🔥 Climate Correlation Analysis")

corr = filtered.corr(numeric_only=True)

fig = go.Figure(
    data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu"
    )
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 🔮 ML MODEL: TEMPERATURE PREDICTION
# -----------------------------
st.subheader("🔮 Climate Prediction Model (2050 Temperature)")

features = [
    "co2_concentration_ppm",
    "sea_level_rise_mm",
    "heatwave_days",
    "drought_index",
    "forest_cover_percent",
    "renewable_energy_share",
    "air_quality_index"
]

X = df[features]
y = df["predicted_temperature_2050"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

st.write("Model R² Score:", round(model.score(X_test, y_test), 3))

# -----------------------------
# SINGLE PREDICTION INPUT
# -----------------------------
st.subheader("🎯 Predict Future Temperature")

input_data = []

for col in features:
    val = st.number_input(col, float(df[col].min()), float(df[col].max()))
    input_data.append(val)

if st.button("Predict"):
    prediction = model.predict([input_data])
    st.success(f"Predicted Temperature (2050): {round(prediction[0], 2)} °C")

# -----------------------------
# DOWNLOAD BUTTON
# -----------------------------
st.markdown("---")

csv = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Filtered Data",
    csv,
    "climate_data.csv",
    "text/csv"
)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("🚀 Built for Decode Labs Internship | Advanced Climate Intelligence Dashboard | Ilyas Ahmad khan Lodhi")