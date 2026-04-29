import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px

def naive_predict(y_train, y_test):
    predictions = []
    current_data = y_train.copy()
    for i in range(len(y_test)):
        pred = current_data[-1] if len(current_data) > 0 else np.mean(current_data)
        predictions.append(pred)
        current_data = np.append(current_data, y_test[i])
    return np.array(predictions)

def arima_predict(y_train, y_test):
    from pmdarima import auto_arima
    predictions = []
    current_data = y_train.copy()
    for i in range(len(y_test)):
        try:
            model = auto_arima(current_data, seasonal=False, trace=False, suppress_warnings=True)
            pred = model.predict(n_periods=1)[0]
        except:
            pred = np.mean(current_data) if len(current_data) > 0 else 0
        predictions.append(pred)
        current_data = np.append(current_data, y_test[i])
    return np.array(predictions)

def lgbm_predict(y_train, y_test, lag=5):
    import lightgbm as lgb
    predictions = []
    current_data = y_train.copy()
    
    def create_lagged_features(data, lag):
        df = pd.DataFrame(data, columns=['y'])
        for i in range(1, lag + 1):
            df[f'lag_{i}'] = df['y'].shift(i)
        return df.dropna()
    
    for i in range(len(y_test)):
        if len(current_data) > lag:
            df = create_lagged_features(current_data, lag)
            X_train = df.drop('y', axis=1).values
            y_train_data = df['y'].values
            
            model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            model.fit(X_train, y_train_data)
            
            last_lags = current_data[-lag:]
            pred = model.predict(last_lags.reshape(1, -1))[0]
        else:
            pred = np.mean(current_data) if len(current_data) > 0 else 0
        predictions.append(pred)
        current_data = np.append(current_data, y_test[i])
    return np.array(predictions)

def predict_next_value_ensemble(models_weights, y):
    predictions = []
    for model_name, weight in models_weights.items():
        if model_name == 'Naive':
            pred = y[-1] if len(y) > 0 else np.mean(y)
        elif model_name == 'ARIMA':
            from pmdarima import auto_arima
            try:
                model = auto_arima(y, seasonal=False, trace=False, suppress_warnings=True)
                pred = model.predict(n_periods=1)[0]
            except:
                pred = np.mean(y) if len(y) > 0 else 0
        elif model_name == 'LightGBM':
            import lightgbm as lgb
            lag = min(5, len(y) // 2)
            if len(y) > lag:
                df = pd.DataFrame(y, columns=['y'])
                for i in range(1, lag + 1):
                    df[f'lag_{i}'] = df['y'].shift(i)
                df = df.dropna()
                X_train = df.drop('y', axis=1).values
                y_train_data = df['y'].values
                model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                model.fit(X_train, y_train_data)
                last_lags = y[-lag:]
                pred = model.predict(last_lags.reshape(1, -1))[0]
            else:
                pred = np.mean(y) if len(y) > 0 else 0
        else:
            pred = np.mean(y) if len(y) > 0 else 0
        predictions.append((pred, weight))
    
    ensemble_pred = sum(pred * weight for pred, weight in predictions)
    return ensemble_pred

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

def detect_outliers_zscore(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    outliers = np.abs(z_scores) > threshold
    return outliers, np.sum(outliers)

def detect_outliers_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers, np.sum(outliers)

def detect_outliers_percentile(data, lower_percent=0.01, upper_percent=0.99):
    lower_bound = np.percentile(data, lower_percent * 100)
    upper_bound = np.percentile(data, upper_percent * 100)
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers, np.sum(outliers)

st.set_page_config(page_title="Time Series Forecasting", layout="wide")

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'has_date' not in st.session_state:
    st.session_state.has_date = False
if 'dates' not in st.session_state:
    st.session_state.dates = None
if 'metrics_df' not in st.session_state:
    st.session_state.metrics_df = None
if 'weights' not in st.session_state:
    st.session_state.weights = None
if 'ensemble_preds' not in st.session_state:
    st.session_state.ensemble_preds = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'test_dates' not in st.session_state:
    st.session_state.test_dates = None

page = st.sidebar.selectbox('Select Page', ['Data Validation', 'Model Prediction', 'Backtesting & Conclusion'])

if page == 'Data Validation':
    st.title('🔍 Data Validation')
    
    uploaded_file = st.file_uploader('Upload Excel File (.xlsx)', type='xlsx')
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=0)
            
            if 'y' not in df.columns:
                st.error('❌ Missing required column "y"')
                st.stop()
            
            y = df['y'].values
            
            if np.any(pd.isna(y)):
                st.error('❌ Column "y" contains missing values')
                st.stop()
            
            if not np.issubdtype(y.dtype, np.number):
                st.error('❌ Column "y" must contain numeric values')
                st.stop()
            
            has_date = 'date' in df.columns
            dates = df['date'].tolist() if has_date else None
            
            st.session_state.uploaded_file = uploaded_file
            st.session_state.df = df
            st.session_state.y = y
            st.session_state.has_date = has_date
            st.session_state.dates = dates
            
            st.success('✅ Data validation passed!')
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric('Total Samples', len(y))
            with col2:
                st.metric('Has Date Column', has_date)
            
            st.subheader('📊 Outlier Detection')
            z_outliers, z_count = detect_outliers_zscore(y)
            iqr_outliers, iqr_count = detect_outliers_iqr(y)
            pct_outliers, pct_count = detect_outliers_percentile(y)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Z-score Outliers', z_count)
            with col2:
                st.metric('IQR Outliers', iqr_count)
            with col3:
                st.metric('Percentile Outliers', pct_count)
            
            if z_count > 0 or iqr_count > 0 or pct_count > 0:
                st.warning('⚠️ Potential outliers detected. Consider reviewing before proceeding.')
            
            st.subheader('📋 Data Preview (Last 10 Samples)')
            preview_df = df.tail(10).copy()
            if has_date:
                preview_df['date'] = preview_df['date'].astype(str)
            st.dataframe(preview_df, use_container_width=True)
            
            with st.spinner('🔄 Running backtesting analysis...'):
                train_size = int(len(y) * 0.8)
                y_train = y[:train_size]
                y_test = y[train_size:]
                test_dates = dates[train_size:] if has_date else None
                
                naive_preds = naive_predict(y_train, y_test)
                arima_preds = arima_predict(y_train, y_test)
                lgbm_preds = lgbm_predict(y_train, y_test)
                
                models = ['Naive', 'ARIMA', 'LightGBM']
                predictions = [naive_preds, arima_preds, lgbm_preds]
                
                metrics = []
                for model, preds in zip(models, predictions):
                    rmse, mae, mape = calculate_metrics(y_test, preds)
                    metrics.append({
                        'Model': model,
                        'RMSE': rmse,
                        'MAE': mae,
                        'MAPE': mape
                    })
                
                metrics_df = pd.DataFrame(metrics)
                weights = 1 / metrics_df['RMSE'].values
                weights = weights / weights.sum()
                metrics_df['Weight'] = weights
                
                ensemble_preds = np.average(predictions, axis=0, weights=weights)
                
                st.session_state.metrics_df = metrics_df
                st.session_state.weights = weights
                st.session_state.ensemble_preds = ensemble_preds
                st.session_state.y_test = y_test
                st.session_state.test_dates = test_dates
            
            st.success('✅ Backtesting completed! You can now view results in other pages.')
        
        except Exception as e:
            st.error(f'❌ Error: {str(e)}')

elif page == 'Model Prediction':
    st.title('🔮 Model Prediction')
    
    if st.session_state.y is None:
        st.warning('⚠️ Please upload a dataset first in the Data Validation page.')
    else:
        y = st.session_state.y
        has_date = st.session_state.has_date
        dates = st.session_state.dates
        
        if st.session_state.metrics_df is not None:
            weights_dict = dict(zip(st.session_state.metrics_df['Model'], st.session_state.metrics_df['Weight']))
            next_pred = predict_next_value_ensemble(weights_dict, y)
            
            st.subheader('Next Step Forecast (y_{N+1})')
            st.markdown(f'<div style="font-size: 48px; font-weight: bold; color: #1e88e5; text-align: center; padding: 20px;">{next_pred:.6f}</div>', unsafe_allow_html=True)
            
            st.subheader('⚖️ Model Weights')
            st.dataframe(st.session_state.metrics_df[['Model', 'Weight']].style.format({'Weight': '{:.4f}'}), use_container_width=True)
            
            if has_date:
                last_date = pd.Timestamp(dates[-1])
                next_date = last_date + pd.Timedelta(days=1)
                st.info(f'Last date in dataset: {last_date.strftime("%Y-%m-%d")}')
                st.info(f'Forecast for: {next_date.strftime("%Y-%m-%d")}')

elif page == 'Backtesting & Conclusion':
    st.title('📊 Backtesting & Conclusion')
    
    if st.session_state.metrics_df is None:
        st.warning('⚠️ Please upload a dataset first in the Data Validation page.')
    else:
        metrics_df = st.session_state.metrics_df
        ensemble_preds = st.session_state.ensemble_preds
        y_test = st.session_state.y_test
        test_dates = st.session_state.test_dates
        has_date = st.session_state.has_date
        
        st.subheader('📋 Model Performance Metrics')
        st.dataframe(metrics_df.style.format({
            'RMSE': '{:.6f}',
            'MAE': '{:.6f}',
            'MAPE': '{:.2f}%',
            'Weight': '{:.4f}'
        }), use_container_width=True)
        
        st.subheader('📈 RMSE Comparison')
        fig_rmse = px.bar(metrics_df, x='Model', y='RMSE', color='Model',
                        title='RMSE by Model', labels={'RMSE': 'Root Mean Squared Error'})
        st.plotly_chart(fig_rmse, use_container_width=True)
        
        st.subheader('📈 MAE Comparison')
        fig_mae = px.bar(metrics_df, x='Model', y='MAE', color='Model',
                       title='MAE by Model', labels={'MAE': 'Mean Absolute Error'})
        st.plotly_chart(fig_mae, use_container_width=True)
        
        st.subheader('📈 MAPE Comparison')
        fig_mape = px.bar(metrics_df, x='Model', y='MAPE', color='Model',
                        title='MAPE by Model', labels={'MAPE': 'Mean Absolute Percentage Error (%)'})
        st.plotly_chart(fig_mape, use_container_width=True)
        
        ensemble_rmse, ensemble_mae, ensemble_mape = calculate_metrics(y_test, ensemble_preds)
        
        st.subheader('🏆 Ensemble Prediction Results')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('RMSE', f'{ensemble_rmse:.6f}')
        with col2:
            st.metric('MAE', f'{ensemble_mae:.6f}')
        with col3:
            st.metric('MAPE', f'{ensemble_mape:.2f}%')
        
        result_df = pd.DataFrame({'y': ensemble_preds})
        if has_date and test_dates:
            result_df.insert(0, 'date', test_dates)
        
        st.subheader('📊 Backtest Forecasts Preview')
        preview_df = result_df.copy()
        if has_date:
            preview_df['date'] = preview_df['date'].astype(str)
        st.dataframe(preview_df, use_container_width=True)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Forecasts')
            metrics_df.to_excel(writer, index=False, sheet_name='Metrics')
        output.seek(0)
        
        st.download_button(
            label='📥 Download Results',
            data=output,
            file_name='forecasting_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
        st.subheader('📝 Conclusion')
        best_model = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']
        st.success(f'Best performing model: **{best_model}**')
        st.write('The ensemble forecast combines predictions from multiple models weighted by their performance, providing a more robust prediction than any single model.')