import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
warnings.filterwarnings("ignore")


class TimeSeriesAnalysis:
    def __init__(self, data):
        self.data = data
        self.ts_data = None
        self.daily_counts = None
        self.train = None
        self.test = None
        self.model_results = None
        
    def preprocess_data(self):
        self.data['time'] = pd.to_datetime(self.data[['year', 'month', 'day', 'hour']])
        self.ts_data = self.data.set_index('time')
        self.daily_counts = self.ts_data['count'].resample('D').sum().dropna()
        
    def plot_cycles(self):
        daily_cycle = self.ts_data.groupby(self.ts_data.index.hour)['count'].mean()
        weekly_cycle = self.ts_data.groupby(self.ts_data.index.dayofweek)['count'].mean()
        weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        rolling_daily = self.daily_counts.rolling(window=7).mean()

        fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

        axs[0].plot(daily_cycle.index, daily_cycle.values, marker='o', color='blue')
        axs[0].set_title('Daily Cycle – Average Ride Count by Hour')
        axs[0].set_xlabel('Hour of Day')
        axs[0].set_ylabel('Avg Count')
        axs[0].grid(True)

        axs[1].plot(weekly_cycle.index, weekly_cycle.values, marker='o', color='green')
        axs[1].set_title('Weekly Cycle – Average Ride Count by Day of Week')
        axs[1].set_xlabel('Day of Week')
        axs[1].set_ylabel('Avg Count')
        axs[1].set_xticks(range(7))
        axs[1].set_xticklabels(weekday_labels)
        axs[1].grid(True)

        axs[2].plot(self.daily_counts.index, self.daily_counts.values, color='lightgray', label='Daily Count')
        axs[2].plot(rolling_daily.index, rolling_daily.values, color='red', label='7-Day Rolling Mean')
        axs[2].set_title('Seasonal/Yearly Cycle – Smoothed Daily Ride Count')
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Total Count')
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

    def perform_stationarity_check(self):
        result_orig = adfuller(self.daily_counts.dropna())
        print('ADF Statistic (Original):', result_orig[0])
        print('p-value:', result_orig[1])

        diff_1 = self.daily_counts.diff().dropna()
        result_diff1 = adfuller(diff_1)
        print('\nADF Statistic (1st Difference):', result_diff1[0])
        print('p-value:', result_diff1[1])

        diff_seasonal = self.daily_counts.diff(7).dropna()
        result_seasonal = adfuller(diff_seasonal)
        print('\nADF Statistic (7-day Seasonal Difference):', result_seasonal[0])
        print('p-value:', result_seasonal[1])

    def split_data(self):
        self.train = self.daily_counts[:-30]
        self.test = self.daily_counts[-30:]

    def fit_sarimax_model(self):
        model = SARIMAX(
            self.train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_results = model.fit(disp=False)

    def forecast_and_plot(self):
        forecast = self.model_results.get_forecast(steps=30)
        forecast_mean = forecast.predicted_mean
        pred_ci = forecast.conf_int()

        mse_loss = (self.test - forecast_mean) ** 2

        differenced = self.daily_counts.diff().dropna()
        residuals = self.model_results.resid

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

        axes[0].plot(self.daily_counts[-90:], label='Observed', color='black')
        forecast_mean.plot(ax=axes[0], label='Forecast', color='red')
        axes[0].fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.3)
        axes[0].set_title('SARIMAX Forecast – Last 30 Days')
        axes[0].set_ylabel('Ride Count')
        axes[0].legend()

        axes[1].plot(differenced[-90:], color='purple', label='1st Order Differenced')
        axes[1].axhline(0, linestyle='--', color='gray', linewidth=1)
        axes[1].set_title('Differenced Series – Stationarity Check')
        axes[1].set_ylabel('Δ Ride Count')
        axes[1].legend()

        axes[2].plot(mse_loss.index, mse_loss.values, color='orange', marker='o', label='Squared Error (MSE)')
        axes[2].set_title('Forecast Error – Squared Loss (MSE)')
        axes[2].set_ylabel('MSE')
        axes[2].set_xlabel('Date')
        axes[2].legend()

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(residuals)
        plt.title('Residuals of the SARIMAX Model')
        plt.show()

        plot_acf(residuals, lags=30)
        plt.title('ACF of Residuals')
        plt.show()

        plot_pacf(residuals, lags=30)
        plt.title('PACF of Residuals')
        plt.show()

    def run(self):
        self.preprocess_data()
        self.plot_cycles()
        self.perform_stationarity_check()
        self.split_data()
        self.fit_sarimax_model()
        self.forecast_and_plot()



data = pd.read_csv('../data/ola_updated.csv')


tsa = TimeSeriesAnalysis(data)


tsa.run()
