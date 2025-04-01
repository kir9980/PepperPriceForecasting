import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import io
import openpyxl
from datetime import datetime

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Pepper Price Prediction", layout="wide")

st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stSelectbox, .stFileUploader {margin-bottom: 1rem;}
    .stDataFrame {width: 100%;}
    .metric-box {border-radius: 0.5rem; padding: 1rem; background-color: #f0f2f6;}
    .plot-container {margin-top: 2rem;}
    .success-box {background-color: #e6f7e6; padding: 1rem; border-radius: 0.5rem;}
    .developer-credit {
        position: fixed;
        bottom: 10px;
        right: 10px;
        background-color: rgba(255,255,255,0.7);
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Pepper Price Prediction and Forecasting")
st.markdown("""
    This app predicts pepper prices using linear regression. Upload your data, select districts, 
    and get forecasts with performance metrics.
    """)

st.markdown('<div class="developer-credit">Developed by Kiran Basavannappagowda - M.Tech in AI, REVA University (RACE)</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Input Parameters")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"], 
                                   help="Upload your Excel file with price data")
    
    if uploaded_file is not None:
        df_data = pd.read_excel(uploaded_file, sheet_name="Sheet1")
        unique_districts = df_data['District'].unique().tolist()
        
        selected_districts = st.multiselect(
            "Select Districts", 
            options=unique_districts,
            default=unique_districts[:3] if len(unique_districts) >= 3 else unique_districts,
            help="Select districts to include in analysis"
        )
        
        months = st.slider("Test Data Period (months)", 1, 12, 6, 
                          help="Select number of months for test period")
        
        forecast_days = st.slider("Forecast Period (days)", 1, 30, 15,
                                help="Select number of days for future forecast")
        
        st.markdown("---")
        st.markdown("**Note:** The app will:")
        st.markdown("- Remove outliers using IQR")
        st.markdown("- Fill missing values")
        st.markdown(f"- Generate {forecast_days}-day forecast")
        st.markdown("- Provide performance metrics")

if uploaded_file is not None and selected_districts:
    try:
        df_data = pd.read_excel(uploaded_file, sheet_name="Sheet1")
        df_data["Date"] = pd.to_datetime(df_data["Date"]).dt.strftime("%m-%d-%y")

        df_copy = df_data[df_data['District'].isin(selected_districts)]

        def remove_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        filtered_df = remove_outliers(df_copy, 'Max')
        filtered_df.reset_index(drop=True, inplace=True)

        filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
        grouped_df = filtered_df.groupby('Date').agg(
            Max=('Max', 'mean'),
            Min=('Min', 'mean'),
            Modal=('Modal', 'mean'),
            Sum_Arrival=('Arrivals', 'mean')
        ).reset_index()

        grouped_df.set_index('Date', inplace=True)
        date_range = pd.date_range(start=grouped_df.index.min(), end=grouped_df.index.max())
        grouped_df = grouped_df.reindex(date_range, fill_value=np.nan)
        grouped_df.fillna(method='ffill', inplace=True)
        grouped_df.reset_index(inplace=True)
        grouped_df.rename(columns={'index': 'Date'}, inplace=True)

        def mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        def evaluate_performance(y_test, y_pred, model_name):
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            return mae, mse, rmse, mape

        df = grouped_df.copy()

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        end_date = df.index.max()
        test_start_date = end_date - pd.DateOffset(months=months)
        df = df[df.index <= end_date]

        df['Smoothed_Max'] = df['Max'].rolling(window=7, min_periods=1).mean()

        for i in range(7):
            df[f'Weight_Day{i+1}'] = df['Max'].shift(i) / df['Smoothed_Max'].shift(i)

        df['Weight_Avg'] = df[[f'Weight_Day{i+1}' for i in range(7)]].mean(axis=1)
        df.dropna(inplace=True)

        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month

        for lag in range(1, 5):
            df[f'Lag{lag}'] = df['Smoothed_Max'].shift(lag)

        df.dropna(inplace=True)

        df_train = df[df.index < test_start_date]
        df_test = df[df.index >= test_start_date]

        features = ['DayOfWeek', 'Month', 'Lag1', 'Lag2', 'Lag3', 'Lag4']
        target = 'Smoothed_Max'

        X_train, y_train = df_train[features], df_train[target]
        X_test, y_test = df_test[features], df_test[target]

        mlr = LinearRegression()
        mlr.fit(X_train, y_train)

        df_test['Smoothed_Predicted_Max'] = mlr.predict(X_test)

        df_test['Weighted_Predicted_Max'] = df_test['Smoothed_Predicted_Max'] * df_test['Weight_Avg']

        mae, mse, rmse, mape = evaluate_performance(df_test['Smoothed_Max'], df_test['Smoothed_Predicted_Max'], "Optimized MLR")

        forecast_start = end_date + pd.DateOffset(days=1)
        forecast_end = forecast_start + pd.DateOffset(days=forecast_days-1)

        forecast_dates = pd.date_range(start=forecast_start, end=forecast_end)
        forecast_values = []

        last_values = df.iloc[-1][['Lag1', 'Lag2', 'Lag3', 'Lag4']].values

        for next_date in forecast_dates:
            next_features = np.array([next_date.dayofweek, next_date.month] + list(last_values)).reshape(1, -1)
            next_pred = mlr.predict(next_features)[0]
            last_values = np.roll(last_values, 1)
            last_values[0] = next_pred
            forecast_values.append(next_pred)

        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted_Smoothed_Max': forecast_values})
        forecast_df.set_index('Date', inplace=True)

        latest_weight_avg = df['Weight_Avg'].iloc[-7:].values
        forecast_df['Actual_Forecasted_Max'] = forecast_df['Forecasted_Smoothed_Max'] * latest_weight_avg.mean()

        test_results = df_test[['Max', 'Weighted_Predicted_Max']].copy()
        test_results['Difference'] = abs(test_results['Max'] - test_results['Weighted_Predicted_Max'])
        test_results['Error_Percentage'] = (test_results['Difference'] / test_results['Max']) * 100
        
        test_results.reset_index(inplace=True)
        test_results.rename(columns={
            'Max': 'Actual_Max_Price',
            'Weighted_Predicted_Max': 'Predicted_Actual_Max',
            'Difference': 'Absolute_Difference',
            'Error_Percentage': 'Error_%'
        }, inplace=True)

        forecast_output = forecast_df.reset_index()[['Date', 'Actual_Forecasted_Max']]
        forecast_output.rename(columns={'Actual_Forecasted_Max': 'Forecasted_Price'}, inplace=True)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            test_results[['Date', 'Actual_Max_Price', 'Predicted_Actual_Max', 'Absolute_Difference', 'Error_%']].to_excel(
                writer, sheet_name="Test_Results", index=False)
            
            forecast_output.to_excel(writer, sheet_name=f"{forecast_days}_Day_Forecast", index=False)
            
            metrics_df = pd.DataFrame({
                'Metric': ['MAE', 'MSE', 'RMSE', 'MAPE'],
                'Value': [mae, mse, rmse, mape]
            })
            metrics_df.to_excel(writer, sheet_name="Performance_Metrics", index=False)
        output.seek(0)

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Visualizations", "🔍 Data", "📥 Download"])
        
        with tab1:
            st.subheader("Model Performance Metrics")
            cols = st.columns(4)
            with cols[0]:
                st.metric("Mean Absolute Error", f"{mae:.2f}")
            with cols[1]:
                st.metric("Mean Squared Error", f"{mse:.2f}")
            with cols[2]:
                st.metric("Root Mean Squared Error", f"{rmse:.2f}")
            with cols[3]:
                st.metric("Mean Absolute % Error", f"{mape:.2f}%")
            
        
        with tab2:
            st.subheader("Price Trends")
            
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(df_test.index, df_test['Smoothed_Max'], label='Actual Smoothed Max Price', marker='o', color='#1f77b4')
            ax1.plot(df_test.index, df_test['Smoothed_Predicted_Max'], 
                    label='Smoothed Predicted Max Price', linestyle='dashed', marker='x', color='#ff7f0e')
            ax1.axvline(x=test_start_date, color='gray', linestyle='--', label='Test Start')
            ax1.legend()
            ax1.set_title("Actual vs Predicted Smoothed Max Prices")
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(df_test.index, df_test['Max'], label='Actual Max Price', marker='o', color='#1f77b4')
            ax2.plot(df_test.index, df_test['Weighted_Predicted_Max'], 
                    label='Predicted Max Price', linestyle='dashed', marker='x', color='#ff7f0e')
            ax2.axvline(x=test_start_date, color='gray', linestyle='--', label='Test Start')
            ax2.legend()
            ax2.set_title("Actual vs Predicted Max Prices")
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(df.index, df['Max'], label='Historical Prices', color='#1f77b4')
            ax3.plot(forecast_df.index, forecast_df['Actual_Forecasted_Max'], 
                    label='Forecasted Prices', marker='o', linestyle='dashed', color='#2ca02c')
            ax3.axvline(x=end_date, color='gray', linestyle='--', label='Forecast Start')
            ax3.legend()
            ax3.set_title(f"Historical Prices with {forecast_days}-Day Forecast")
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Price')
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
        
        with tab3:
            st.subheader("Test Results Data")
            st.dataframe(test_results.style.format({
                'Actual_Max_Price': '{:.2f}',
                'Predicted_Actual_Max': '{:.2f}',
                'Absolute_Difference': '{:.2f}',
                'Error_%': '{:.2f}'
            }))
            
            st.subheader(f"{forecast_days}-Day Forecast Data")
            st.dataframe(forecast_output.style.format({'Forecasted_Price': '{:.2f}'}))
        
        with tab4:
            st.subheader("Download Results")
            st.markdown(f"""
                Click the button below to download the complete analysis results in Excel format.
                The file contains:
                - Test results with actual vs predicted values
                - {forecast_days}-day price forecast
                - Performance metrics
            """)
            
            st.download_button(
                label="📥 Download Excel Report",
                data=output,
                file_name=f"Pepper_Price_Forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download the complete analysis results"
            )
            
            st.markdown("---")
            st.markdown("**Analysis Parameters:**")
            st.markdown(f"- Selected Districts: {', '.join(selected_districts)}")
            st.markdown(f"- Test Period: {months} months")
            st.markdown(f"- Forecast Period: {forecast_days} days")
            st.markdown(f"- Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        st.markdown("---")
        st.success("✅ Analysis completed successfully!")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.warning("Please check your input file format and try again.")
else:
    st.info("ℹ️ Please upload an Excel file and select districts to begin analysis")
