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

# Filter data based on date range
if len(date_range) == 2:
    filtered_df = df[(df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])]
else:
    filtered_df = df.copy()

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
    st.markdown(f'''<div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);">
        st.markdown(f'<div class="metric-value">{filtered_df["Temperature"].mean():.1f}°C</div>', unsafe_allow_html=True)
        <div class="metric-label">Avg Temperature</div>
    </div>''', unsafe_allow_html=True)

with mcol2:
    st.markdown(f'''<div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <div class="metric-value">{filtered_df["Humidity"].mean():.0f}%</div>
        <div class="metric-label">Avg Humidity</div>
    </div>''', unsafe_allow_html=True)

with mcol3:
    st.markdown(f'''<div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
        <div class="metric-value">{filtered_df["WindSpeed"].mean():.1f}</div>
        <div class="metric-label">Avg Wind (km/h)</div>
    </div>''', unsafe_allow_html=True)

with mcol4:
    rainy_days = (filtered_df["Rainfall"] > 0).sum()
    st.markdown(f'''<div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <div class="metric-value">{rainy_days}</div>
        <div class="metric-label">Rainy Days</div>
    </div>''', unsafe_allow_html=True)

with mcol5:
    total_rain = filtered_df["Rainfall"].sum()
    st.markdown(f'''<div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
        <div class="metric-value">{total_rain:.1f}mm</div>
        <div class="metric-label">Total Rainfall</div>
    </div>''', unsafe_allow_html=True)

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
    
    st.markdown("#### Weekly Temperature Heatmap")
    filtered_df["Week"] = filtered_df["Date"].dt.isocalendar().week
    filtered_df["Weekday"] = filtered_df["Date"].dt.day_name()
    weekly_temp = filtered_df.groupby(["Week", "Weekday"])["Temperature"].mean().reset_index()
    pivot_temp = weekly_temp.pivot(index="Weekday", columns="Week", values="Temperature")
    
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot_temp = pivot_temp.reindex([d for d in weekday_order if d in pivot_temp.index])
    
    fig = px.imshow(pivot_temp, color_continuous_scale="RdYlBu_r",
                   title="Temperature Heatmap by Week & Day",
                   labels=dict(color="Temp (°C)"))
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.markdown("#### Humidity Over Time")
        fig = px.area(filtered_df, x="Date", y="Humidity",
                     color_discrete_sequence=["#1e88e5"],
                     title="Humidity Area Chart")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_d:
        st.markdown("#### Rainfall Events")
        rain_df = filtered_df[filtered_df["Rainfall"] > 0].copy()
        if len(rain_df) > 0:
            fig = px.bar(rain_df, x="Date", y="Rainfall", color="Temperature",
                        color_continuous_scale="YlOrRd",
                        title="Rainfall Events (Colored by Temperature)")
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rainfall data in selected range.")
    
    st.markdown("#### Correlation Matrix")
    corr_cols = ["Temperature", "Humidity", "WindSpeed", "Rainfall"]
    corr = filtered_df[corr_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                   zmin=-1, zmax=1,
                   title="Weather Variable Correlations")
    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("#### Complete Weather Dataset")
    
    display_df = filtered_df.copy()
    display_df["Temp_Category"] = display_df["Temperature"].apply(
        lambda x: "🔥 Hot" if x > 32 else "☀️ Warm" if x > 27 else "🌤 Mild" if x > 23 else "❄️ Cool"
    )
    display_df["Rain_Status"] = display_df["Rainfall"].apply(
        lambda x: "🌧 Rainy" if x > 5 else "💧 Light" if x > 0 else "☀️ Dry"
    )
    
    col_e, col_f = st.columns([3, 1])
    with col_e:
        search = st.text_input("🔍 Search by date (YYYY-MM-DD)")
    with col_f:
        sort_by = st.selectbox("Sort by", ["Date", "Temperature", "Humidity", "Rainfall"])
    
    if search:
        display_df = display_df[display_df["Date"].astype(str).str.contains(search)]
    
    display_df = display_df.sort_values(sort_by, ascending=(sort_by == "Date"))
    
    st.dataframe(display_df[["Date", "Temperature", "Humidity", "WindSpeed", "Rainfall", "Temp_Category", "Rain_Status"]],
                use_container_width=True, height=500)
    
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="weather_data_filtered.csv",
        mime="text/csv"
    )

with tab5:
    st.markdown("#### 🤖 Model Performance & Insights")
    
    X_full = df[["Humidity", "WindSpeed", "Rainfall"]]
    y_true = df["Temperature"]
    y_pred = model.predict(X_full)
    
    col_g, col_h = st.columns(2)
    
    with col_g:
        st.markdown("#### Actual vs Predicted")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                                marker=dict(color="#1e88e5", size=8, opacity=0.7),
                                name="Predictions"))
        fig.add_trace(go.Scatter(x=[y_true.min(), y_true.max()],
                                y=[y_true.min(), y_true.max()],
                                mode="lines", line=dict(color="#e53935", dash="dash"),
                                name="Perfect Prediction"))
        fig.update_layout(
            xaxis_title="Actual Temperature (°C)",
            yaxis_title="Predicted Temperature (°C)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_h:
        st.markdown("#### Prediction Error Distribution")
        errors = y_true - y_pred
        fig = px.histogram(x=errors, nbins=20, color_discrete_sequence=["#43a047"],
                          title="Residual Error Distribution",
                          labels={"x": "Error (Actual - Predicted)"})
        fig.update_layout(template="plotly_white", height=400)
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    
    st.markdown("#### Model Metrics")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("R² Score", f"{r2:.3f}")
    mc2.metric("RMSE", f"{rmse:.2f}°C")
    mc3.metric("MAE", f"{mae:.2f}°C")
    mc4.metric("Model Type", "Linear Regression")
    
    st.markdown("#### Feature Importance (Coefficients)")
    coef_df = pd.DataFrame({
        "Feature": ["Humidity", "Wind Speed", "Rainfall"],
        "Coefficient": model.coef_,
        "Impact": ["Decreases Temp" if c < 0 else "Increases Temp" for c in model.coef_]
    })
    fig = px.bar(coef_df, x="Feature", y="Coefficient", color="Impact",
                color_discrete_map={"Decreases Temp": "#e53935", "Increases Temp": "#43a047"},
                title="How Each Feature Affects Temperature Prediction")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"📌 **Insight:** The model equation is: **Temperature = {model.intercept_:.2f} + ({model.coef_[0]:.3f} × Humidity) + ({model.coef_[1]:.3f} × WindSpeed) + ({model.coef_[2]:.3f} × Rainfall)**")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>🌦 Weather Forecast Dashboard | Built with Streamlit & Plotly | 100 Days Dataset</div>", unsafe_allow_html=True)
