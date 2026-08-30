import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Weather Forecast Dashboard",
    page_icon="🌤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e88e5, #43a047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    return pickle.load(open("weather_model.pkl", "rb"))

model = load_model()

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/weather.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.strftime('%B')
    df['Day'] = df['Date'].dt.day
    df['Week'] = df['Date'].dt.isocalendar().week
    return df

df = load_data()

# Header
st.markdown('<div class="main-header">🌦 Weather Forecast Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">100 Days of Weather Intelligence | Predict • Analyze • Visualize</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    st.markdown("---")
    
    st.subheader("🔮 Temperature Predictor")
    humidity = st.slider("💧 Humidity (%)", 0, 100, 60, help="Current humidity level")
    wind = st.slider("💨 Wind Speed (km/h)", 0, 50, 10, help="Wind speed in km/h")
    rain = st.slider("🌧 Rainfall (mm)", 0.0, 30.0, 5.0, 0.1, help="Rainfall amount in mm")
    
    st.markdown("---")
    
    st.subheader("📅 Date Range Filter")
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.date_input("Select Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    st.markdown("---")
    st.info("💡 **Tip:** Use the sliders to predict temperature for custom weather conditions!")

# Filter data safely based on date range selection
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    filtered_df = df[(df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])]
else:
    filtered_df = df.copy()

# Prevent calculations if the filtered date range yields zero results
if filtered_df.empty:
    st.error("⚠️ No data available for the selected date range. Please choose a different range.")
    st.stop()

# Prediction Section
st.markdown("### 🔮 Live Temperature Prediction")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.metric("💧 Humidity", f"{humidity}%")
with col2:
    st.metric("💨 Wind Speed", f"{wind} km/h")
with col3:
    st.metric("🌧 Rainfall", f"{rain} mm")

prediction = model.predict([[humidity, wind, rain]])[0]

# Color based on temperature
temp_color = "#e53935" if prediction > 35 else "#fb8c00" if prediction > 30 else "#43a047" if prediction > 25 else "#1e88e5"
temp_label = "🔥 Very Hot" if prediction > 35 else "☀️ Hot" if prediction > 30 else "🌤 Pleasant" if prediction > 25 else "❄️ Cool"

st.markdown(f'''
<div class="prediction-box" style="background: linear-gradient(135deg, {temp_color}dd 0%, {temp_color} 100%);">
    <div style="font-size: 1.2rem; opacity: 0.9;">Predicted Temperature</div>
    <div style="font-size: 4rem; font-weight: 800;">{prediction:.1f}°C</div>
    <div style="font-size: 1rem; opacity: 0.9;">{temp_label}</div>
</div>
''", unsafe_allow_html=True)

st.markdown("---")

# Key Metrics
st.markdown("### 📊 Key Weather Metrics")
mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)

with mcol1:
    st.markdown(f'''
    <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);">
        <div class="metric-value">{filtered_df["Temperature"].mean():.1f}°C</div>
        <div class="metric-label">Avg Temperature</div>
    </div>
    ''', unsafe_allow_html=True)

with mcol2:
    st.markdown(f'''
    <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <div class="metric-value">{filtered_df["Humidity"].mean():.0f}%</div>
        <div class="metric-label">Avg Humidity</div>
    </div>
    ''', unsafe_allow_html=True)

with mcol3:
    st.markdown(f'''
    <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
        <div class="metric-value">{filtered_df["WindSpeed"].mean():.1f}</div>
        <div class="metric-label">Avg Wind (km/h)</div>
    </div>
    ''', unsafe_allow_html=True)

with mcol4:
    rainy_days = (filtered_df["Rainfall"] > 0).sum()
    st.markdown(f'''
    <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <div class="metric-value">{rainy_days}</div>
        <div class="metric-label">Rainy Days</div>
    </div>
    ''', unsafe_allow_html=True)

with mcol5:
    total_rain = filtered_df["Rainfall"].sum()
    st.markdown(f'''
    <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
        <div class="metric-value">{total_rain:.1f}mm</div>
        <div class="metric-label">Total Rainfall</div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")

# Tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trend Analysis", "🌡 Temperature Deep Dive", "💧 Humidity & Rain", "📋 Data Table", "🔬 Model Insights"])

with tab1:
    st.markdown("#### Weather Trends Over Time")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Temperature Trend", "Humidity Trend", "Wind Speed Trend", "Rainfall Pattern"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    fig.add_trace(go.Scatter(
        x=filtered_df["Date"], y=filtered_df["Temperature"],
        mode="lines+markers", name="Temperature",
        line=dict(color="#e53935", width=2),
        marker=dict(size=5)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=filtered_df["Date"], y=filtered_df["Humidity"],
        mode="lines+markers", name="Humidity",
        line=dict(color="#1e88e5", width=2),
        marker=dict(size=5)
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=filtered_df["Date"], y=filtered_df["WindSpeed"],
        mode="lines+markers", name="Wind Speed",
        line=dict(color="#43a047", width=2),
        marker=dict(size=5)
    ), row=2, col=1)
    
    colors = ["#4fc3f7" if r == 0 else "#0288d1" for r in filtered_df["Rainfall"]]
    fig.add_trace(go.Bar(
        x=filtered_df["Date"], y=filtered_df["Rainfall"],
        name="Rainfall", marker_color=colors
    ), row=2, col=2)
    
    fig.update_layout(height=700, showlegend=False, template="plotly_white")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### Temperature Distribution")
        fig = px.histogram(filtered_df, x="Temperature", nbins=20,
                          color_discrete_sequence=["#e53935"],
                          marginal="box",
                          title="Temperature Distribution")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.markdown("#### Temperature vs Humidity")
        fig = px.scatter(filtered_df, x="Humidity", y="Temperature",
                        size="WindSpeed", color="Rainfall",
                        color_continuous_scale="Blues",
                        title="Temperature vs Humidity (Size = Wind, Color = Rain)",
                        hover_data=["Date"])
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("#### 💧 Humidity & Rain Analysis")
    fig3 = px.line(filtered_df, x="Date", y=["Humidity", "Rainfall"], 
                   title="Humidity and Rainfall System Trends Over Time",
                   color_discrete_map={"Humidity": "#1e88e5", "Rainfall": "#764ba2"})
    fig3.update_layout(template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.markdown("#### 📋 Filtered Historical Data Records")
    st.dataframe(filtered_df, use_container_width=True)

with tab5:
    st.markdown("#### 🔬 Predictive Model Insights")
    st.info("The application calculates live inferences using a serialized backend machine learning model pipeline.")
    st.write("### Model Training Information & Target Features:")
    st.write("- **Features Evaluated:** Humidity, WindSpeed, Rainfall")
    st.write("- **Pipeline Origin File:** `weather_model.pkl` via `pickle` system integration.")
