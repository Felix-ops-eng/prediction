# AAPL Stock Price Forecast Analysis

A Streamlit web application for analyzing and forecasting AAPL stock prices using various time series models.

## Features

- **Data Overview**: Historical stock price visualization
- **Mean Model**: Simple forecast using training data mean
- **Naive Model**: Forecast using last observed value with prediction intervals
- **Seasonal Naive**: Seasonal forecasting with 20 trading day cycle
- **Residual Analysis**: ACF plots, histograms, and Q-Q plots
- **Error Metrics**: RMSE, MAE, MAPE, quantile scores, and Winkler score

## Installation & Usage

### Local Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

### Streamlit Community Cloud Deployment

1. Ensure your GitHub repository only contains:
   - `app.py` (主应用文件)
   - `requirements.txt` (依赖包)
   - `README.md` (本文件)

2. Remove any other Python files (like `interactive_macd_plot.py`, `aapl_mean_model.py`, etc.) from the repository root

3. Push to GitHub:
   ```bash
   git add app.py requirements.txt README.md .gitignore
   git commit -m "Deploy Streamlit app"
   git push origin main
   ```

4. Deploy on [share.streamlit.io](https://share.streamlit.io/):
   - **Main file path**: `app.py`
   - **Repository**: Your GitHub repo
   - **Branch**: main/master

## Project Structure

```
your-repo/
├── app.py               # Main application (主文件)
├── requirements.txt      # Dependencies
├── .gitignore           # Files to ignore
└── README.md            # This file
```

## Models Implemented

1. **Mean Model**: Forecast = training data mean
2. **Naive Model**: Forecast = last observed value
3. **Seasonal Naive**: Forecast = last value from same season (20 trading days)

## Metrics

- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- MAPE: Mean Absolute Percentage Error
- Quantile Scores (Pinball Loss)
- Winkler Score for prediction intervals

## Technologies

- Streamlit
- Pandas
- NumPy
- Plotly
- Scipy
- Statsmodels
- Akshare (for data)

## License

MIT
