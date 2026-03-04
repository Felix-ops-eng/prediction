import streamlit as st
from modules.data_download import download_apple_stock_data, download_stocks_2024_2025, download_last_month_data, load_data
from modules.data_analysis import calculate_statistics, generate_correlation_matrix, calculate_rolling_correlation, detect_anomalies
from modules.investment_models import generate_investment_advice, markowitz_model, risk_parity_model
from modules.data_quality import check_data_quality
import warnings

# 设置中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title='股票分析与投资建议系统',
    page_icon='📈',
    layout='wide'
)

# 主应用
st.title('股票分析与投资建议系统')

# 侧边栏
st.sidebar.header('功能选择')
analysis_option = st.sidebar.selectbox(
    '选择分析类型',
    ['数据概览', '统计参数', '相关性分析', '异常值检测', '投资建议', '回测分析', 'Apple 股票数据', '2024-2025 股票数据', '数据质量检查']
)

# 加载数据
data = load_data()

if data is not None:
    if analysis_option == '数据概览':
        st.header('数据概览')
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('数据基本信息')
            st.write(f"数据行数: {data.shape[0]}")
            st.write(f"数据列数: {data.shape[1]}")
            st.write(f"股票数量: {data['symbol'].nunique()}")
            st.write(f"日期范围: {data['date'].min()} 至 {data['date'].max()}")
        
        with col2:
            st.subheader('股票列表')
            st.write(data['symbol'].unique())
        
        st.subheader('数据样本')
        st.dataframe(data.head())
    
    elif analysis_option == '统计参数':
        st.header('股票统计参数')
        
        stats_df = calculate_statistics(data)
        
        st.dataframe(stats_df.round(4))
        
        # 可视化统计参数
        st.subheader('统计参数可视化')
        
        # 夏普比率柱状图
        st.subheader('夏普比率')
        fig, ax = plt.subplots(figsize=(12, 6))
        import seaborn as sns
        sns.barplot(x='symbol', y='sharpe_ratio', data=stats_df, ax=ax)
        ax.set_title('各股票夏普比率')
        ax.set_ylabel('夏普比率')
        st.pyplot(fig)
        
        # 波动率柱状图
        st.subheader('波动率')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='symbol', y='std', data=stats_df, ax=ax)
        ax.set_title('各股票波动率')
        ax.set_ylabel('波动率')
        st.pyplot(fig)
    
    elif analysis_option == '相关性分析':
        st.header('相关性分析')
        
        # 相关性矩阵热力图
        correlation_matrix = generate_correlation_matrix(data)
        
        st.subheader('相关性矩阵热力图')
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
        ax.set_title('股票收益率相关性矩阵')
        st.pyplot(fig)
        
        # 滚动相关性
        st.subheader('滚动相关性分析')
        avg_rolling_corr = calculate_rolling_correlation(data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(avg_rolling_corr, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
        ax.set_title('平均滚动相关性（30天窗口）')
        st.pyplot(fig)
    
    elif analysis_option == '异常值检测':
        st.header('异常值检测')
        
        anomaly_df = detect_anomalies(data)
        stats_df = calculate_statistics(data)
        
        # 合并异常值检测结果和统计信息
        import pandas as pd
        anomaly_info = pd.merge(anomaly_df, stats_df[['symbol', 'sharpe_ratio', 'std']], on='symbol')
        
        st.dataframe(anomaly_info.round(4))
        
        # 异常值数量可视化
        st.subheader('异常值数量')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='symbol', y='anomaly_count', data=anomaly_df, ax=ax)
        ax.set_title('各股票异常值数量（Z-score > 3）')
        st.pyplot(fig)
    
    elif analysis_option == '投资建议':
        st.header('投资建议')
        
        # 模型选择 - 默认使用马科维兹模型（预测成功率通常最高）
        model_option = st.selectbox(
            '选择投资模型',
            ['马科维兹均值-方差模型', '夏普比率模型', '风险平价模型'],
            index=0
        )
        
        if model_option == '夏普比率模型':
            stats_df = calculate_statistics(data)
            correlation_matrix = generate_correlation_matrix(data)
            advice = generate_investment_advice(stats_df, correlation_matrix)
            
            st.subheader('投资配置建议')
            st.dataframe(advice.round(4))
            
            # 投资比例热力图
            st.subheader('投资比例热力图')
            fig, ax = plt.subplots(figsize=(12, 2))
            sns.heatmap(advice[['weight']].T, annot=True, cmap='Greens', ax=ax, fmt='.2%')
            ax.set_xticklabels(advice['symbol'])
            ax.set_title('投资资金分配比例（夏普比率模型）')
            st.pyplot(fig)
            
            # 投资建议文本
            st.subheader('投资策略建议')
            top_stocks = advice.nlargest(3, 'sharpe_ratio')['symbol'].tolist()
            low_risk_stocks = advice.nsmallest(3, 'volatility')['symbol'].tolist()
            
            st.write(f"**推荐重点关注股票**: {', '.join(top_stocks)}")
            st.write(f"**低风险稳健股票**: {', '.join(low_risk_stocks)}")
            st.write("**投资策略**: 基于夏普比率的优化配置，兼顾收益与风险。")
        
        elif model_option == '马科维兹均值-方差模型':
            st.subheader('马科维兹均值-方差模型')
            
            # 计算马科维兹模型
            advice, optimal_portfolio = markowitz_model(data)
            
            st.subheader('最优投资组合配置')
            st.dataframe(advice.round(4))
            
            # 投资比例热力图
            st.subheader('投资比例热力图')
            fig, ax = plt.subplots(figsize=(12, 2))
            sns.heatmap(advice[['weight']].T, annot=True, cmap='Greens', ax=ax, fmt='.2%')
            ax.set_xticklabels(advice['symbol'])
            ax.set_title('投资资金分配比例（马科维兹模型）')
            st.pyplot(fig)
            
            # 最优投资组合信息
            st.subheader('最优投资组合信息')
            st.write(f"**预期年化收益率**: {optimal_portfolio['return']:.2%}")
            st.write(f"**预期波动率**: {optimal_portfolio['volatility']:.2%}")
            st.write(f"**夏普比率**: {optimal_portfolio['sharpe_ratio']:.4f}")
            
            # 投资建议文本
            st.subheader('投资策略建议')
            top_stocks = advice.nlargest(3, 'weight')['symbol'].tolist()
            st.write(f"**重点配置股票**: {', '.join(top_stocks)}")
            st.write("**投资策略**: 基于均值-方差优化的投资组合，追求风险调整后收益最大化。")
        
        elif model_option == '风险平价模型':
            st.subheader('风险平价模型')
            
            # 计算风险平价模型
            advice = risk_parity_model(data)
            
            st.subheader('风险平价配置')
            st.dataframe(advice.round(4))
            
            # 投资比例热力图
            st.subheader('投资比例热力图')
            fig, ax = plt.subplots(figsize=(12, 2))
            sns.heatmap(advice[['weight']].T, annot=True, cmap='Greens', ax=ax, fmt='.2%')
            ax.set_xticklabels(advice['symbol'])
            ax.set_title('投资资金分配比例（风险平价模型）')
            st.pyplot(fig)
            
            # 投资建议文本
            st.subheader('投资策略建议')
            low_vol_stocks = advice.nlargest(3, 'weight')['symbol'].tolist()
            st.write(f"**重点配置股票**: {', '.join(low_vol_stocks)}")
            st.write("**投资策略**: 基于风险平价的配置，每个资产贡献相等的风险。")
    
    elif analysis_option == '回测分析':
        st.header('回测分析')
        
        # 下载上个月的数据
        last_month_data = download_last_month_data()
        
        if not last_month_data.empty:
            st.subheader('上个月数据概览')
            st.write(f"数据行数: {last_month_data.shape[0]}")
            st.write(f"日期范围: {last_month_data['date'].min()} 至 {last_month_data['date'].max()}")
            
            # 计算上个月的收益率
            last_month_returns = {}
            for symbol in last_month_data['symbol'].unique():
                stock_data = last_month_data[last_month_data['symbol'] == symbol]['close']
                if len(stock_data) > 1:
                    return_rate = (stock_data.iloc[-1] / stock_data.iloc[0] - 1) * 100
                    last_month_returns[symbol] = return_rate
            
            # 显示上个月收益率
            if last_month_returns:
                returns_df = pd.DataFrame(list(last_month_returns.items()), columns=['symbol', 'monthly_return'])
                returns_df = returns_df.sort_values('monthly_return', ascending=False)
                
                st.subheader('上个月收益率')
                st.dataframe(returns_df.round(2))
                
                # 可视化上个月收益率
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(x='symbol', y='monthly_return', data=returns_df, ax=ax)
                ax.set_title('各股票上个月收益率（%）')
                ax.axhline(y=0, color='red', linestyle='--')
                st.pyplot(fig)
                
                # 模型比较
                st.subheader('模型预测成功率比较')
                
                # 1. 夏普比率模型
                stats_df = calculate_statistics(data)
                top_sharpe = stats_df.nlargest(5, 'sharpe_ratio')['symbol'].tolist()
                
                # 2. 马科维兹模型
                markowitz_advice, _ = markowitz_model(data)
                top_markowitz = markowitz_advice.nlargest(5, 'weight')['symbol'].tolist()
                
                # 3. 风险平价模型
                risk_parity_advice = risk_parity_model(data)
                top_risk_parity = risk_parity_advice.nlargest(5, 'weight')['symbol'].tolist()
                
                # 计算各模型的预测成功率
                models = {
                    '夏普比率模型': top_sharpe,
                    '马科维兹模型': top_markowitz,
                    '风险平价模型': top_risk_parity
                }
                
                success_rates = {}
                for model_name, top_stocks in models.items():
                    successful = 0
                    for symbol in top_stocks:
                        if symbol in last_month_returns and last_month_returns[symbol] > 0:
                            successful += 1
                    success_rate = (successful / len(top_stocks)) * 100 if top_stocks else 0
                    success_rates[model_name] = success_rate
                
                # 显示模型比较结果
                comparison_df = pd.DataFrame(
                    list(success_rates.items()),
                    columns=['模型', '预测成功率']
                )
                comparison_df = comparison_df.sort_values('预测成功率', ascending=False)
                
                st.dataframe(comparison_df.round(2))
                
                # 可视化模型比较
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(x='模型', y='预测成功率', data=comparison_df, ax=ax)
                ax.set_title('各模型预测成功率比较')
                ax.set_ylabel('成功率 (%)')
                ax.set_ylim(0, 100)
                st.pyplot(fig)
                
                # 显示各模型预测的股票
                st.subheader('各模型预测的股票')
                for model_name, top_stocks in models.items():
                    st.write(f"**{model_name}**: {', '.join(top_stocks)}")
                
                # 计算各模型的平均收益率
                model_returns = {}
                for model_name, top_stocks in models.items():
                    model_return = 0
                    count = 0
                    for symbol in top_stocks:
                        if symbol in last_month_returns:
                            model_return += last_month_returns[symbol]
                            count += 1
                    avg_return = model_return / count if count > 0 else 0
                    model_returns[model_name] = avg_return
                
                # 显示各模型的平均收益率
                returns_df = pd.DataFrame(
                    list(model_returns.items()),
                    columns=['模型', '平均收益率']
                )
                returns_df = returns_df.sort_values('平均收益率', ascending=False)
                
                st.subheader('各模型平均收益率')
                st.dataframe(returns_df.round(2))
                
                # 可视化模型平均收益率
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(x='模型', y='平均收益率', data=returns_df, ax=ax)
                ax.set_title('各模型平均收益率比较')
                ax.set_ylabel('平均收益率 (%)')
                ax.axhline(y=0, color='red', linestyle='--')
                st.pyplot(fig)
        else:
            st.warning('无法获取上个月的数据进行回测分析')
    
    elif analysis_option == 'Apple 股票数据':
        st.header('Apple 股票数据分析')
        
        # 日期选择
        from datetime import datetime, timedelta
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('开始日期', value=datetime.now() - timedelta(days=365))
        with col2:
            end_date = st.date_input('结束日期', value=datetime.now())
        
        # 下载 Apple 股票数据
        if st.button('下载 Apple 股票数据'):
            apple_data = download_apple_stock_data(start_date, end_date)
            
            if not apple_data.empty:
                st.subheader('Apple 股票数据概览')
                st.write(f"数据行数: {apple_data.shape[0]}")
                st.write(f"日期范围: {apple_data['date'].min()} 至 {apple_data['date'].max()}")
                
                # 显示数据样本
                st.subheader('数据样本')
                st.dataframe(apple_data.head())
                
                # 计算统计参数
                apple_stats = calculate_statistics(apple_data)
                st.subheader('Apple 股票统计参数')
                st.dataframe(apple_stats.round(4))
                
                # 可视化股价走势
                st.subheader('Apple 股价走势')
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(apple_data['date'], apple_data['close'])
                ax.set_title('Apple 股票收盘价走势')
                ax.set_xlabel('日期')
                ax.set_ylabel('收盘价')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # 计算并显示收益率
                apple_data['return'] = apple_data['close'].pct_change() * 100
                st.subheader('Apple 股票收益率')
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(apple_data['date'], apple_data['return'])
                ax.set_title('Apple 股票日收益率')
                ax.set_xlabel('日期')
                ax.set_ylabel('收益率 (%)')
                ax.axhline(y=0, color='red', linestyle='--')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # 保存数据为 CSV
                csv_data = apple_data.to_csv(index=False)
                st.download_button(
                    label="下载 Apple 股票数据 CSV",
                    data=csv_data,
                    file_name="apple_stock_data.csv",
                    mime="text/csv"
                )
            else:
                st.warning('无法获取 Apple 股票数据')
    
    elif analysis_option == '2024-2025 股票数据':
        st.header('2024-2025 股票数据下载')
        
        # 下载数据按钮
        if st.button('下载 2024.11 至 2025.11 股票数据'):
            with st.spinner('正在下载数据...'):
                stocks_data = download_stocks_2024_2025()
                
                if not stocks_data.empty:
                    st.subheader('数据概览')
                    st.write(f"数据行数: {stocks_data.shape[0]}")
                    st.write(f"日期范围: {stocks_data['date'].min()} 至 {stocks_data['date'].max()}")
                    st.write(f"包含股票数量: {stocks_data['symbol'].nunique()}")
                    st.write(f"包含股票: {', '.join(stocks_data['symbol'].unique())}")
                    
                    # 显示数据样本
                    st.subheader('数据样本')
                    st.dataframe(stocks_data.head())
                    
                    # 保存数据为 CSV
                    csv_data = stocks_data.to_csv(index=False)
                    st.download_button(
                        label="下载 2024-2025 股票数据 CSV",
                        data=csv_data,
                        file_name="multiple_stocks_data_2024_2025.csv",
                        mime="text/csv"
                    )
                    
                    # 保存数据到文件
                    stocks_data.to_csv('multiple_stocks_data_2024_2025.csv', index=False)
                    st.success('数据已保存到 multiple_stocks_data_2024_2025.csv 文件')
                else:
                    st.warning('无法获取股票数据')
    
    elif analysis_option == '数据质量检查':
        st.header('数据质量检查')
        
        # 运行数据质量检查
        quality_report = check_data_quality(data)
        
        # 显示数据形状
        st.subheader('数据形状')
        st.write(f"行数: {quality_report['shape'][0]}")
        st.write(f"列数: {quality_report['shape'][1]}")
        
        # 显示缺失值
        st.subheader('缺失值检查')
        missing_df = pd.DataFrame.from_dict(quality_report['missing_values'], orient='index', columns=['缺失值数量'])
        st.dataframe(missing_df)
        
        # 显示基本统计信息
        st.subheader('基本统计信息')
        st.dataframe(quality_report['basic_statistics'])
        
        # 显示股票信息
        if 'unique_symbols' in quality_report:
            st.subheader('股票信息')
            st.write(f"股票数量: {quality_report['unique_symbols']}")
            st.write(f"股票列表: {', '.join(quality_report['symbols'])}")
        
        # 显示日期范围
        if 'date_range' in quality_report:
            st.subheader('日期范围')
            st.write(f"开始日期: {quality_report['date_range']['start']}")
            st.write(f"结束日期: {quality_report['date_range']['end']}")
        
        # 显示重复日期检查
        if 'duplicate_dates' in quality_report:
            st.subheader('重复日期检查')
            st.write(f"重复日期数量: {quality_report['duplicate_dates']}")
            if quality_report['duplicate_dates'] > 0:
                st.warning('数据中存在重复日期，可能需要进一步处理')
            else:
                st.success('数据中没有重复日期')
        
        # 数据质量总结
        st.subheader('数据质量总结')
        total_missing = sum(quality_report['missing_values'].values())
        if total_missing == 0:
            st.success('数据质量良好，没有缺失值')
        else:
            st.warning(f'数据中存在 {total_missing} 个缺失值，需要处理')
else:
    st.error('无法加载数据，请检查数据文件是否存在')

# 页脚
st.sidebar.markdown('---')
st.sidebar.markdown('📊 股票分析与投资建议系统')
st.sidebar.markdown('基于历史数据的量化分析')