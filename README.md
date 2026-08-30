# 🌦 Weather Forecast Dashboard - Enhanced (100 Days)

An enhanced weather prediction and visualization dashboard built with Streamlit, featuring a **100-day dataset** and interactive Plotly charts.

## ✨ Features

- 🔮 **Live Temperature Prediction** — Predict temperature from humidity, wind speed, and rainfall
- 📊 **Interactive Visualizations** — Plotly charts with hover tooltips and zoom
- 📅 **Date Range Filtering** — Filter data by custom date ranges
- 📈 **Trend Analysis** — Multi-metric time series charts
- 🌡 **Temperature Deep Dive** — Distribution, scatter plots, weekly heatmaps
- 💧 **Humidity & Rain Analysis** — Area charts, rainfall events, correlation matrix
- 📋 **Data Explorer** — Sortable, searchable table with CSV download
- 🔬 **Model Insights** — Actual vs predicted, error distribution, feature importance

## 📁 Dataset

- **100 days** of weather data (2025-01-01 to 2025-04-10)
- Columns: Date, Temperature, Humidity, WindSpeed, Rainfall

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🌐 Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and deploy!

## 📂 Project Structure

```
pbl_enhanced/
├── data/
│   └── weather.csv          # 100-day weather dataset
├── app.py                   # Enhanced Streamlit app
├── weather_model.pkl        # Trained Linear Regression model
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🤖 Model Details

- **Type**: Linear Regression
- **Features**: Humidity, WindSpeed, Rainfall
- **Target**: Temperature
- **Training Data**: 100 days of synthetic weather data

## 📊 Sample Data

| Date       | Temperature | Humidity | WindSpeed | Rainfall |
|------------|-------------|----------|-----------|----------|
| 2025-01-01 | 26.2°C      | 64%      | 13.8 km/h | 0.0 mm   |
| 2025-01-02 | 22.1°C      | 71%      | 14.8 km/h | 17.3 mm  |
| ...        | ...         | ...      | ...       | ...      |
| 2025-04-10 | 32.3°C      | 36%      | 15.1 km/h | 0.0 mm   |
