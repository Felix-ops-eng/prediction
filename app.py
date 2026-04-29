import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.stattools import acf
import akshare as ak

st.set_page_config(page_title='AAPL Stock Forecast Analysis', layout='wide')

@st.cache_data
def load_data():
    df = ak.stock_us_daily(symbol='AAPL')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

def calculate_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def calculate_mape(actual, predicted):
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def quantile_score(actual, quantile_pred, q):
    error = actual - quantile_pred
    score = np.where(error >= 0, (1 - q) * error, q * (-error))
    return np.mean(score)

def winkler_score(actual, lower, upper, alpha=0.05):
    width = upper - lower
    below = actual < lower
    above = actual > upper
    score = width.copy()
    score[below] += (2 / alpha) * (lower[below] - actual[below])
    score[above] += (2 / alpha) * (actual[above] - upper[above])
    return np.mean(score)

df = load_data()

train_ratio = 0.8
split_idx = int(len(df) * train_ratio)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

test_actual = test['close'].values
mean_forecast = train['close'].mean()
naive_forecast = train['close'].iloc[-1]

m = 20
seasonal_indices = np.arange(len(train)) % m
last_season_values = []
for s in range(m):
    season_mask = (seasonal_indices == s)
    if np.any(season_mask):
        last_value = train['close'].iloc[season_mask].iloc[-1]
    else:
        last_value = train['close'].iloc[-1]
    last_season_values.append(last_value)

seasonal_forecasts = np.array([last_season_values[i % m] for i in range(len(test))])
mean_predictions = np.full(len(test), mean_forecast)
naive_predictions = np.full(len(test), naive_forecast)

first_diffs = np.diff(train['close'])
diff_std = np.sqrt(np.var(first_diffs, ddof=1))
t_critical = stats.t.ppf(0.975, df=len(train) - 1)

test_horizons = np.arange(1, len(test) + 1)
interval_radii = t_critical * diff_std * np.sqrt(test_horizons)
lower_bounds = naive_forecast - interval_radii
upper_bounds = naive_forecast + interval_radii

st.title('📈 AAPL Stock Price Forecast Analysis')
st.markdown('---')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'Data Overview',
    'Mean Model',
    'Naive Model', 
    'Seasonal Naive',
    'Residual Analysis',
    'Error Metrics'
])

with tab1:
    st.subheader('📊 Dataset Overview')
    col1, col2, col3 = st.columns(3)
    col1.metric('Total Records', len(df))
    col2.metric('Training Set', len(train))
    col3.metric('Test Set', len(test))
    
    st.subheader('Date Range')
    st.write(f"From: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    st.subheader('Sample Data')
    st.dataframe(df.tail(10), use_container_width=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='AAPL Close Price'))
    fig.update_layout(title='AAPL Historical Stock Price', xaxis_title='Date', yaxis_title='Price ($)', height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader('🔵 Mean Model')
    st.write(f"**Forecast**: ${mean_forecast:.2f} (sample mean of training data)")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test['date'], y=test['close'], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train['date'], y=train['close'], mode='lines', name='Training', line=dict(color='gray', dash='dot')))
    fig.add_hline(y=mean_forecast, line_dash='dash', line_color='red', annotation_text=f'Mean: ${mean_forecast:.2f}')
    fig.update_layout(title='Mean Model Forecast', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    rmse = calculate_rmse(test_actual, mean_predictions)
    mae = calculate_mae(test_actual, mean_predictions)
    mape = calculate_mape(test_actual, mean_predictions)
    
    col1, col2, col3 = st.columns(3)
    col1.metric('RMSE', f'${rmse:.2f}')
    col2.metric('MAE', f'${mae:.2f}')
    col3.metric('MAPE', f'{mape:.2f}%')

with tab3:
    st.subheader('🔴 Naive Model')
    st.write(f"**Forecast**: ${naive_forecast:.2f} (last observed value)")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test['date'], y=test['close'], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train['date'], y=train['close'], mode='lines', name='Training', line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=test['date'], y=upper_bounds, mode='lines', name='Upper PI', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=test['date'], y=lower_bounds, mode='lines', name='Lower PI', line=dict(color='green', dash='dash'), fill='tonexty'))
    fig.add_hline(y=naive_forecast, line_dash='dash', line_color='red', annotation_text=f'Naive: ${naive_forecast:.2f}')
    fig.update_layout(title='Naive Model with 95% Prediction Intervals', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    rmse = calculate_rmse(test_actual, naive_predictions)
    mae = calculate_mae(test_actual, naive_predictions)
    mape = calculate_mape(test_actual, naive_predictions)
    
    col1, col2, col3 = st.columns(3)
    col1.metric('RMSE', f'${rmse:.2f}')
    col2.metric('MAE', f'${mae:.2f}')
    col3.metric('MAPE', f'{mape:.2f}%')

with tab4:
    st.subheader('🟣 Seasonal Naive Model (m=20)')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test['date'], y=test['close'], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train['date'], y=train['close'], mode='lines', name='Training', line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=test['date'], y=seasonal_forecasts, mode='lines', name='Seasonal Forecast', line=dict(color='purple', dash='dash')))
    fig.update_layout(title='Seasonal Naive Model (20 trading day season)', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    rmse = calculate_rmse(test_actual, seasonal_forecasts)
    mae = calculate_mae(test_actual, seasonal_forecasts)
    mape = calculate_mape(test_actual, seasonal_forecasts)
    
    col1, col2, col3 = st.columns(3)
    col1.metric('RMSE', f'${rmse:.2f}')
    col2.metric('MAE', f'${mae:.2f}')
    col3.metric('MAPE', f'{mape:.2f}%')

with tab5:
    st.subheader('🔍 Residual Analysis')
    
    naive_residuals = first_diffs
    
    tab5_1, tab5_2, tab5_3 = st.tabs(['ACF Plot', 'Histogram', 'Q-Q Plot'])
    
    with tab5_1:
        n_lags = 40
        acf_values = acf(naive_residuals, nlags=n_lags)
        confidence = 1.96 / np.sqrt(len(naive_residuals))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(n_lags+1), y=acf_values, mode='markers+lines', name='ACF'))
        fig.add_hline(y=confidence, line_color='red', line_dash='dash')
        fig.add_hline(y=-confidence, line_color='red', line_dash='dash')
        fig.update_layout(title='ACF of Naive Residuals', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5_2:
        mu, std = stats.norm.fit(naive_residuals)
        x_range = np.linspace(mu - 4*std, mu + 4*std, 1000)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=naive_residuals, histnorm='probability density', name='Empirical'))
        fig.add_trace(go.Scatter(x=x_range, y=stats.norm.pdf(x_range, mu, std), mode='lines', name='Normal Fit'))
        fig.update_layout(title='Residual Distribution vs Normal', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5_3:
        sorted_residuals = np.sort(naive_residuals)
        n = len(sorted_residuals)
        theoretical_quantiles = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n, mu, std)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_residuals, mode='markers', name='Residuals', marker=dict(size=3)))
        min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
        max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Reference', line=dict(color='red', dash='dash')))
        fig.update_layout(title='Q-Q Plot', height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader('📉 Error Metrics Comparison')
    
    models = ['Mean Model', 'Naive Model', 'Seasonal Naive']
    rmse_vals = [calculate_rmse(test_actual, mean_predictions), 
                 calculate_rmse(test_actual, naive_predictions),
                 calculate_rmse(test_actual, seasonal_forecasts)]
    mae_vals = [calculate_mae(test_actual, mean_predictions),
                calculate_mae(test_actual, naive_predictions),
                calculate_mae(test_actual, seasonal_forecasts)]
    mape_vals = [calculate_mape(test_actual, mean_predictions),
                 calculate_mape(test_actual, naive_predictions),
                 calculate_mape(test_actual, seasonal_forecasts)]
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=('RMSE', 'MAE', 'MAPE'))
    fig.add_trace(go.Bar(x=models, y=rmse_vals, marker_color=['#2E86AB', '#A23B72', '#F18F01']), row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=mae_vals, marker_color=['#2E86AB', '#A23B72', '#F18F01']), row=1, col=2)
    fig.add_trace(go.Bar(x=models, y=mape_vals, marker_color=['#2E86AB', '#A23B72', '#F18F01']), row=1, col=3)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader('Quantile & Winkler Scores')
    
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    q_scores = [quantile_score(test_actual, naive_forecast, q) for q in quantiles]
    
    fig_q = go.Figure()
    fig_q.add_trace(go.Bar(x=[f'{int(q*100)}%' for q in quantiles], y=q_scores))
    fig_q.update_layout(title='Quantile Scores (Pinball Loss)', height=300)
    st.plotly_chart(fig_q, use_container_width=True)
    
    winkler = winkler_score(test_actual, lower_bounds, upper_bounds)
    st.metric('Winkler Score (95% PI)', f'{winkler:.2f}')

st.markdown('---')
st.write('Data source: akshare | Analysis by Streamlit')