```python
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Weather Forecast Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main Title */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e88e5, #43a047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* Subtitle */
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Metric Cards */
    .metric-card {
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Prediction Box */
    .prediction-box {
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin-top: 10px;
        margin-bottom: 20px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def load_model():
    try:
        with open("weather_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model

    except FileNotFoundError:
        st.error("❌ Model file 'weather_model.pkl' not found.")
        st.stop()


model = load_model()


# ============================================================
# LOAD WEATHER DATA
# ============================================================

@st.cache_data
def load_data():

    try:
        df = pd.read_csv("data/weather.csv")

        # Convert Date column
        df["Date"] = pd.to_datetime(df["Date"])

        # Create additional useful columns
        df["Month"] = df["Date"].dt.strftime("%B")
        df["Day"] = df["Date"].dt.day
        df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

        return df

    except FileNotFoundError:
        st.error("❌ Weather dataset not found at 'data/weather.csv'")
        st.stop()


df = load_data()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-header">🌦️ Weather Forecast Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'''
    <div class="sub-header">
        {len(df)} Days of Weather Intelligence |
        Predict • Analyze • Visualize
    </div>
    ''',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:

    st.header("⚙️ Dashboard Controls")

    st.markdown("---")

    # Prediction Section
    st.subheader("🔮 Temperature Predictor")

    humidity = st.slider(
        "💧 Humidity (%)",
        min_value=0,
        max_value=100,
        value=60,
        help="Select current humidity level"
    )

    wind = st.slider(
        "💨 Wind Speed (km/h)",
        min_value=0,
        max_value=50,
        value=10,
        help="Select wind speed"
    )

    rain = st.slider(
        "🌧️ Rainfall (mm)",
        min_value=0.0,
        max_value=30.0,
        value=5.0,
        step=0.1,
        help="Select rainfall amount"
    )

    st.markdown("---")

    # Date Filter
    st.subheader("📅 Date Range Filter")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    st.markdown("---")

    st.info(
        "💡 **Tip:** Adjust the weather parameters above "
        "to generate a live temperature prediction."
    )


# ============================================================
# FILTER DATA BY DATE
# ============================================================

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    filtered_df = df[
        (df["Date"] >= start_date) &
        (df["Date"] <= end_date)
    ].copy()

else:
    filtered_df = df.copy()


# Safety Check
if filtered_df.empty:

    st.warning(
        "⚠️ No weather data available for the selected date range."
    )

    st.stop()


# ============================================================
# LIVE TEMPERATURE PREDICTION
# ============================================================

st.markdown("## 🔮 Live Temperature Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="💧 Humidity",
        value=f"{humidity}%"
    )

with col2:
    st.metric(
        label="💨 Wind Speed",
        value=f"{wind} km/h"
    )

with col3:
    st.metric(
        label="🌧️ Rainfall",
        value=f"{rain} mm"
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

# Create DataFrame with feature names
# This avoids sklearn feature-name warnings

input_data = pd.DataFrame(
    [[humidity, wind, rain]],
    columns=[
        "Humidity",
        "WindSpeed",
        "Rainfall"
    ]
)

try:
    prediction = model.predict(input_data)[0]

except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.stop()


# ============================================================
# TEMPERATURE STATUS
# ============================================================

if prediction > 35:
    temp_color = "#e53935"
    temp_label = "🔥 Very Hot"

elif prediction > 30:
    temp_color = "#fb8c00"
    temp_label = "☀️ Hot"

elif prediction > 25:
    temp_color = "#43a047"
    temp_label = "🌤️ Pleasant"

else:
    temp_color = "#1e88e5"
    temp_label = "❄️ Cool"


# ============================================================
# PREDICTION DISPLAY
# ============================================================

st.markdown(
    f"""
    <div class="prediction-box"
         style="background: linear-gradient(
         135deg,
         {temp_color}dd 0%,
         {temp_color} 100%
         );">

        <div style="font-size: 1.2rem; opacity: 0.9;">
            Predicted Temperature
        </div>

        <div style="font-size: 4rem; font-weight: 800;">
            {prediction:.1f}°C
        </div>

        <div style="font-size: 1.2rem; opacity: 0.9;">
            {temp_label}
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("---")


# ============================================================
# KEY WEATHER METRICS
# ============================================================

st.markdown("## 📊 Key Weather Metrics")

mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)


# Average Temperature
with mcol1:

    avg_temp = filtered_df["Temperature"].mean()

    st.markdown(
        f"""
        <div class="metric-card"
        style="background: linear-gradient(
        135deg, #ff6b6b 0%, #ee5a24 100%);">

            <div class="metric-value">
                {avg_temp:.1f}°C
            </div>

            <div class="metric-label">
                Avg Temperature
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# Average Humidity
with mcol2:

    avg_humidity = filtered_df["Humidity"].mean()

    st.markdown(
        f"""
        <div class="metric-card"
        style="background: linear-gradient(
        135deg, #4facfe 0%, #00f2fe 100%);">

            <div class="metric-value">
                {avg_humidity:.0f}%
            </div>

            <div class="metric-label">
                Avg Humidity
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# Average Wind Speed
with mcol3:

    avg_wind = filtered_df["WindSpeed"].mean()

    st.markdown(
        f"""
        <div class="metric-card"
        style="background: linear-gradient(
        135deg, #43e97b 0%, #38f9d7 100%);">

            <div class="metric-value">
                {avg_wind:.1f}
            </div>

            <div class="metric-label">
                Avg Wind (km/h)
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# Rainy Days
with mcol4:

    rainy_days = (
        filtered_df["Rainfall"] > 0
    ).sum()

    st.markdown(
        f"""
        <div class="metric-card"
        style="background: linear-gradient(
        135deg, #667eea 0%, #764ba2 100%);">

            <div class="metric-value">
                {rainy_days}
            </div>

            <div class="metric-label">
                Rainy Days
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# Total Rainfall
with mcol5:

    total_rain = filtered_df["Rainfall"].sum()

    st.markdown(
        f"""
        <div class="metric-card"
        style="background: linear-gradient(
        135deg, #fa709a 0%, #fee140 100%);">

            <div class="metric-value">
                {total_rain:.1f} mm
            </div>

            <div class="metric-label">
                Total Rainfall
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("---")


# ============================================================
# VISUALIZATION TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([

    "📈 Trend Analysis",
    "🌡️ Temperature Deep Dive",
    "💧 Humidity & Rain",
    "📋 Data Table",
    "🔬 Model Insights"

])


# ============================================================
# TAB 1 - WEATHER TRENDS
# ============================================================

with tab1:

    st.markdown("### 📈 Weather Trends Over Time")

    fig = make_subplots(

        rows=2,
        cols=2,

        subplot_titles=(
            "🌡️ Temperature Trend",
            "💧 Humidity Trend",
            "💨 Wind Speed Trend",
            "🌧️ Rainfall Pattern"
        ),

        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )


    # Temperature
    fig.add_trace(

        go.Scatter(
            x=filtered_df["Date"],
            y=filtered_df["Temperature"],

            mode="lines+markers",

            name="Temperature",

            line=dict(
                color="#e53935",
                width=2
            ),

            marker=dict(size=5)

        ),

        row=1,
        col=1
    )


    # Humidity
    fig.add_trace(

        go.Scatter(
            x=filtered_df["Date"],
            y=filtered_df["Humidity"],

            mode="lines+markers",

            name="Humidity",

            line=dict(
                color="#1e88e5",
                width=2
            ),

            marker=dict(size=5)

        ),

        row=1,
        col=2
    )


    # Wind Speed
    fig.add_trace(

        go.Scatter(
            x=filtered_df["Date"],
            y=filtered_df["WindSpeed"],

            mode="lines+markers",

            name="Wind Speed",

            line=dict(
                color="#43a047",
                width=2
            ),

            marker=dict(size=5)

        ),

        row=2,
        col=1
    )


    # Rainfall
    fig.add_trace(

        go.Bar(
            x=filtered_df["Date"],
            y=filtered_df["Rainfall"],

            name="Rainfall",

            marker_color="#0288d1"

        ),

        row=2,
        col=2
    )


    fig.update_layout(

        height=700,

        showlegend=False,

        template="plotly_white",

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )


    fig.update_xaxes(tickangle=45)

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TAB 2 - TEMPERATURE ANALYSIS
# ============================================================

with tab2:

    col_a, col_b = st.columns(2)


    # Temperature Distribution
    with col_a:

        st.markdown("### 🌡️ Temperature Distribution")

        fig_temp = px.histogram(

            filtered_df,

            x="Temperature",

            nbins=20,

            color_discrete_sequence=["#e53935"],

            marginal="box",

            title="Distribution of Temperature"

        )

        fig_temp.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            fig_temp,
            use_container_width=True
        )


    # Temperature vs Humidity
    with col_b:

        st.markdown("### 💧 Temperature vs Humidity")

        fig_scatter = px.scatter(

            filtered_df,

            x="Humidity",

            y="Temperature",

            size="WindSpeed",

            color="Rainfall",

            color_continuous_scale="Blues",

            title="Temperature vs Humidity",

            hover_data=["Date"]

        )

        fig_scatter.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            fig_scatter,
            use_container_width=True
        )


# ============================================================
# TAB 3 - HUMIDITY AND RAIN ANALYSIS
# ============================================================

with tab3:

    st.markdown("### 💧 Humidity & Rainfall Analysis")

    fig3 = px.line(

        filtered_df,

        x="Date",

        y=[
            "Humidity",
            "Rainfall"
        ],

        title="Humidity and Rainfall Trends Over Time",

        color_discrete_map={
            "Humidity": "#1e88e5",
            "Rainfall": "#764ba2"
        }

    )

    fig3.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )


# ============================================================
# TAB 4 - DATA TABLE
# ============================================================

with tab4:

    st.markdown("### 📋 Filtered Historical Weather Records")

    # Display dataset
    st.dataframe(

        filtered_df,

        use_container_width=True,

        hide_index=True

    )


    # Download CSV Button
    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="📥 Download Filtered Weather Data",

        data=csv,

        file_name="filtered_weather_data.csv",

        mime="text/csv"

    )


# ============================================================
# TAB 5 - MODEL INSIGHTS
# ============================================================

with tab5:

    st.markdown("### 🔬 Predictive Model Insights")

    st.info(
        "The application uses a trained Machine Learning model "
        "to predict temperature based on weather conditions."
    )

    st.markdown("### 🧠 Model Input Features")

    col_x, col_y, col_z = st.columns(3)

    with col_x:
        st.metric(
            "💧 Feature 1",
            "Humidity"
        )

    with col_y:
        st.metric(
            "💨 Feature 2",
            "Wind Speed"
        )

    with col_z:
        st.metric(
            "🌧️ Feature 3",
            "Rainfall"
        )


    st.markdown("---")

    st.markdown("### 📁 Model Information")

    st.write(
        "• **Model File:** `weather_model.pkl`"
    )

    st.write(
        "• **Dataset File:** `data/weather.csv`"
    )

    st.write(
        "• **Target Variable:** Temperature"
    )

    st.write(
        "• **Prediction Features:** Humidity, WindSpeed, Rainfall"
    )

    st.success(
        "✅ Model loaded successfully and ready for live predictions!"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:gray; padding:10px;">
        🌦️ Weather Forecast Dashboard |
        Built with Streamlit • Machine Learning • Plotly
    </div>
    """,
    unsafe_allow_html=True
)
```
