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

st.markdown('<div class="developer-credit">Developed by Kiran Basavannappagowda - AI-07 batch, M.Tech in AI, REVA University (RACE)</div>', unsafe_allow_html=True)

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
        df['Smoothed_Min'] = df['Min'].rolling(window=7, min_periods=1).mean()

        for i in range(7):
            df[f'Weight_Day_Max{i+1}'] = df['Max'].shift(i) / df['Smoothed_Max'].shift(i)
            df[f'Weight_Day_Min{i+1}'] = df['Min'].shift(i) / df['Smoothed_Min'].shift(i)

        df['Weight_Avg_Max'] = df[[f'Weight_Day_Max{i+1}' for i in range(7)]].mean(axis=1)
        df['Weight_Avg_Min'] = df[[f'Weight_Day_Min{i+1}' for i in range(7)]].mean(axis=1)
        df.dropna(inplace=True)

        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month

        for lag in range(1, 5):
            df[f'Lag_Max{lag}'] = df['Smoothed_Max'].shift(lag)
            df[f'Lag_Min{lag}'] = df['Smoothed_Min'].shift(lag)

        df.dropna(inplace=True)

        df_train = df[df.index < test_start_date]
        df_test = df[df.index >= test_start_date]

        features_max = ['DayOfWeek', 'Month', 'Lag_Max1', 'Lag_Max2', 'Lag_Max3', 'Lag_Max4']
        features_min = ['DayOfWeek', 'Month', 'Lag_Min1', 'Lag_Min2', 'Lag_Min3', 'Lag_Min4']
        
        target_max = 'Smoothed_Max'
        target_min = 'Smoothed_Min'

        # Split data for Max price prediction
        X_train_max, y_train_max = df_train[features_max], df_train[target_max]
        X_test_max, y_test_max = df_test[features_max], df_test[target_max]

        # Split data for Min price prediction
        X_train_min, y_train_min = df_train[features_min], df_train[target_min]
        X_test_min, y_test_min = df_test[features_min], df_test[target_min]

        # Train Model for Max price prediction
        mlr_max = LinearRegression()
        mlr_max.fit(X_train_max, y_train_max)

        # Train Model for Min price prediction
        mlr_min = LinearRegression()
        mlr_min.fit(X_train_min, y_train_min)
        

        df_test['Smoothed_Predicted_Max'] = mlr_max.predict(X_test_max)
        df_test['Smoothed_Predicted_Min'] = mlr_min.predict(X_test_min)

        df_test['Weighted_Predicted_Max'] = df_test['Smoothed_Predicted_Max'] * df_test['Weight_Avg_Max']
        df_test['Weighted_Predicted_Min'] = df_test['Smoothed_Predicted_Min'] * df_test['Weight_Avg_Min']

        mae_max, mse_max, rmse_max, mape_max = evaluate_performance(df_test['Smoothed_Max'], df_test['Smoothed_Predicted_Max'], "Optimized MLR for Max")
        mae_min, mse_min, rmse_min, mape_min = evaluate_performance(df_test['Smoothed_Min'], df_test['Smoothed_Predicted_Min'], "Optimized MLR for Min")

        forecast_start = end_date + pd.DateOffset(days=1)
        forecast_end = forecast_start + pd.DateOffset(days=forecast_days-1)

        forecast_dates = pd.date_range(start=forecast_start, end=forecast_end)
        forecast_values_max = []
        forecast_values_min = []

        last_values_max = df.iloc[-1][['Lag_Max1', 'Lag_Max2', 'Lag_Max3', 'Lag_Max4']].values
        last_values_min = df.iloc[-1][['Lag_Min1', 'Lag_Min2', 'Lag_Min3', 'Lag_Min4']].values


         
        forecast_data = []

        for next_date in forecast_dates:
            next_features_max = np.array([next_date.dayofweek, next_date.month] + list(last_values_max)).reshape(1, -1)
            next_pred_max = mlr_max.predict(next_features_max)[0]
    
            next_features_min = np.array([next_date.dayofweek, next_date.month] + list(last_values_min)).reshape(1, -1)
            next_pred_min = mlr_min.predict(next_features_min)[0]
    
            last_values_max = np.roll(last_values_max, 1)
            last_values_max[0] = next_pred_max
    
            last_values_min = np.roll(last_values_min, 1)
            last_values_min[0] = next_pred_min
    
            forecast_data.append({
                'Date': next_date,
                'Forecasted_Smoothed_Max': next_pred_max,
                'Forecasted_Smoothed_Min': next_pred_min
            })

        forecast_df = pd.DataFrame(forecast_data)
        forecast_df.reset_index(inplace=True)
        
        
        latest_weight_avg_max = df['Weight_Avg_Max'].iloc[-7:].values
        latest_weight_avg_min = df['Weight_Avg_Min'].iloc[-7:].values
        
        forecast_df['Actual_Forecasted_Max'] = forecast_df['Forecasted_Smoothed_Max'] * latest_weight_avg_max.mean()
        forecast_df['Actual_Forecasted_Min'] = forecast_df['Forecasted_Smoothed_Min'] * latest_weight_avg_min.mean()
        
        test_results = df_test[['Max', 'Weighted_Predicted_Max','Min','Weighted_Predicted_Min']].copy()
        
        test_results['Difference_Actual_Max'] = np.abs(test_results['Max'] - test_results['Weighted_Predicted_Max'])
        test_results['Error_Percentage_Actual_Max'] = (test_results['Difference_Actual_Max'] / test_results['Max']) * 100

        test_results['Difference_Actual_Min'] = np.abs(test_results['Min'] - test_results['Weighted_Predicted_Min'])
        test_results['Error_Percentage_Actual_Min'] = (test_results['Difference_Actual_Min'] / test_results['Min']) * 100
        
        test_results.reset_index(inplace=True)
        test_results.rename(columns={
            'Max': 'Actual_Max_Price',
            'Weighted_Predicted_Max': 'Predicted_Actual_Max',
            'Difference_Actual_Max': 'Absolute_Difference_Max',
            'Error_Percentage_Actual_Max': 'Error_Percentage_Actual_Max_%',
            'Min': 'Actual_Min_Price',
            'Weighted_Predicted_Min': 'Predicted_Actual_Min',
            'Difference_Actual_Min': 'Absolute_Difference_Min',
            'Error_Percentage_Actual_Min': 'Error_Percentage_Actual_Min_%'
        }, inplace=True)
        

        
        forecast_output = forecast_df[['Date','Actual_Forecasted_Max','Actual_Forecasted_Min']].copy()
        forecast_df.set_index('Date', inplace=True)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            test_results[['Date', 'Actual_Max_Price','Predicted_Actual_Max','Absolute_Difference_Max','Error_Percentage_Actual_Max_%','Actual_Min_Price','Predicted_Actual_Min','Absolute_Difference_Min','Error_Percentage_Actual_Min_%']].to_excel(
                writer, sheet_name="Test_Results", index=False)
            
            forecast_output.to_excel(writer, sheet_name=f"{forecast_days}_Day_Forecast", index=False)
            
            metrics_df = pd.DataFrame({
                'Metric': ['MAE_Max', 'MSE_Max', 'RMSE_Max', 'MAPE_Max','MAE_Min', 'MSE_Min', 'RMSE_Min', 'MAPE_Min'],
                'Value': [mae_max, mse_max, rmse_max, mape_max,mae_min, mse_min, rmse_min, mape_min ]
            })
            metrics_df.to_excel(writer, sheet_name="Performance_Metrics", index=False)
        output.seek(0)

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Visualizations", "🔍 Data", "📥 Download"])
        
        with tab1:
            st.subheader("Model Performance Metrics for Max Price")
            cols = st.columns(4)
            with cols[0]:
                st.metric("Mean Absolute Error", f"{mae_max:.2f}")
            with cols[1]:
                st.metric("Mean Squared Error", f"{mse_max:.2f}")
            with cols[2]:
                st.metric("Root Mean Squared Error", f"{rmse_max:.2f}")
            with cols[3]:
                st.metric("Mean Absolute % Error", f"{mape_max:.2f}%")
            
            st.subheader("Model Performance Metrics for Min Price")
            cols = st.columns(4)
            with cols[0]:
                st.metric("Mean Absolute Error", f"{mae_min:.2f}")
            with cols[1]:
                st.metric("Mean Squared Error", f"{mse_min:.2f}")
            with cols[2]:
                st.metric("Root Mean Squared Error", f"{rmse_min:.2f}")
            with cols[3]:
                st.metric("Mean Absolute % Error", f"{mape_min:.2f}%")
        
        with tab2:
            st.subheader("Max Price Trends")
            
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(df_test.index, df_test['Smoothed_Max'], label='Actual Smoothed Max Price', marker='o', color='#1f77b4')
            ax1.plot(df_test.index, df_test['Smoothed_Predicted_Max'], 
                    label='Smoothed Predicted Max Price', linestyle='dashed', marker='x', color='#ff7f0e')
            ax1.axvline(x=test_start_date, color='gray', linestyle='--', label='Test Start')
            ax1.legend()
            ax1.set_title("Actual vs Predicted Smoothed Max Prices")
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Max Price')
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
            ax2.set_ylabel('Max Price')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(df.index, df['Max'], label='Historical Prices', color='#1f77b4')
            ax3.plot(forecast_df.index, forecast_df['Actual_Forecasted_Max'], 
                    label='Forecasted Max Prices', marker='o', linestyle='dashed', color='#2ca02c')
            ax3.axvline(x=end_date, color='gray', linestyle='--', label='Forecast Start')
            ax3.legend()
            ax3.set_title(f"Historical Prices with {forecast_days}-Day Forecast")
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Max Price')
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)

            st.subheader("Min Price Trends")
            
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            ax4.plot(df_test.index, df_test['Smoothed_Min'], label='Actual Smoothed Min Price', marker='o', color='#1f77b4')
            ax4.plot(df_test.index, df_test['Smoothed_Predicted_Min'], 
                    label='Smoothed Predicted Min Price', linestyle='dashed', marker='x', color='#ff7f0e')
            ax4.axvline(x=test_start_date, color='gray', linestyle='--', label='Test Start')
            ax4.legend()
            ax4.set_title("Actual vs Predicted Smoothed Min Prices")
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Min Price')
            ax4.grid(True, alpha=0.3)
            st.pyplot(fig4)

            fig5, ax5 = plt.subplots(figsize=(12, 6))
            ax5.plot(df_test.index, df_test['Min'], label='Actual Min Price', marker='o', color='#1f77b4')
            ax5.plot(df_test.index, df_test['Weighted_Predicted_Min'], 
                    label='Predicted Min Price', linestyle='dashed', marker='x', color='#ff7f0e')
            ax5.axvline(x=test_start_date, color='gray', linestyle='--', label='Test Start')
            ax5.legend()
            ax5.set_title("Actual vs Predicted Min Prices")
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Min Price')
            ax5.grid(True, alpha=0.3)
            st.pyplot(fig5)
            
            fig6, ax6 = plt.subplots(figsize=(12, 6))
            ax6.plot(df.index, df['Min'], label='Historical Min Prices', color='#1f77b4')
            ax6.plot(forecast_df.index, forecast_df['Actual_Forecasted_Min'], 
                    label='Forecasted Min Prices', marker='o', linestyle='dashed', color='#2ca02c')
            ax6.axvline(x=end_date, color='gray', linestyle='--', label='Forecast Start')
            ax6.legend()
            ax6.set_title(f"Historical Min Prices with {forecast_days}-Day Forecast")
            ax6.set_xlabel('Date')
            ax6.set_ylabel('Min Price')
            ax6.grid(True, alpha=0.3)
            st.pyplot(fig6)
       
        with tab3:
            st.subheader("Test Results Data")
            st.dataframe(test_results.style.format({
                'Actual_Max_Price': '{:.2f}',
                'Predicted_Actual_Max': '{:.2f}',
                'Absolute_Difference_Max': '{:.2f}',
                'Error_Percentage_Actual_Max_%': '{:.2f}',
                'Actual_Min_Price': '{:.2f}',
                'Predicted_Actual_Min': '{:.2f}',
                'Absolute_Difference_Min': '{:.2f}',
                'Error_Percentage_Actual_Min_%': '{:.2f}'
            }))
            
            st.subheader(f"{forecast_days}-Day Forecast Data")
            try:
                styled_df = forecast_output.style.format(
                    formatter={
                        'Actual_Forecasted_Max': '{:.2f}',
                        'Actual_Forecasted_Min': '{:.2f}'
                    },
                    subset=['Actual_Forecasted_Max', 'Actual_Forecasted_Min']
                )
                st.dataframe(styled_df)
            except Exception as e:
                st.error(f"Couldn't apply styling: {str(e)}")
        
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
        st.success("✅ Process completed successfully!")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.warning("Please check your input file format and try again.")
else:
    st.info("ℹ️ Please upload an Excel file and select districts to begin analysis")