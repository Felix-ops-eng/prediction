# Time Series Forecasting Web App

A web application for time series forecasting with multiple models comparison.

## Features

- **Data Validation**: Upload Excel files and validate data quality
- **Outlier Detection**: Z-score, IQR, and percentile-based outlier detection
- **Multiple Models**: Naive, ARIMA, and LightGBM models
- **Ensemble Forecasting**: Weighted ensemble prediction based on model performance
- **Backtesting**: 80/20 train-test split with rolling one-step-ahead forecasts
- **Metrics**: RMSE, MAE, and MAPE for model comparison
- **Visualization**: Bar charts for performance metrics comparison

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Input Data Format

The application accepts Excel files (.xlsx) with the following schema:

- **Required column**: `y` - Univariate time series data (numeric)
- **Optional column**: `date` - Date/time information for display

## Output

- **Forecast**: One-step ahead prediction (y_{N+1})
- **Backtest Results**: Excel file with forecasts and model metrics
- **Visualizations**: Bar charts comparing model performance

## Project Structure

```
.
├── app.py          # Main application
├── requirements.txt # Dependencies
├── README.md       # Documentation
└── .gitignore      # Git ignore rules
```

## License

MIT