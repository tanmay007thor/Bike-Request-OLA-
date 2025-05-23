import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def visualize_time_series(ts_data):
    daily_cycle = ts_data.groupby(ts_data.index.hour)['count'].mean()
    weekly_cycle = ts_data.groupby(ts_data.index.dayofweek)['count'].mean()
    daily_counts = ts_data['count'].resample('D').sum()
    rolling_daily = daily_counts.rolling(window=7).mean()

    fig, axs = plt.subplots(3, 1, figsize=(14, 12))

    axs[0].plot(daily_cycle.index, daily_cycle.values, marker='o')
    axs[0].set_title('Daily Cycle')

    axs[1].plot(weekly_cycle.index, weekly_cycle.values, marker='o')
    axs[1].set_title('Weekly Cycle')

    axs[2].plot(daily_counts.index, daily_counts.values, label='Daily')
    axs[2].plot(rolling_daily.index, rolling_daily.values, label='Rolling', color='red')
    axs[2].legend()
    axs[2].set_title('Smoothed Daily Count')

    plt.tight_layout()
    plt.show()

def plot_acf_pacf(daily_counts):
    series_list = [
        ('Original', daily_counts),
        ('1st Diff', daily_counts.diff().dropna()),
        ('7-day Diff', daily_counts.diff(7).dropna()),
        ('Log + Diff', np.log1p(daily_counts).diff().dropna())
    ]

    fig, axs = plt.subplots(len(series_list), 2, figsize=(14, 10))
    for i, (title, series) in enumerate(series_list):
        plot_acf(series, lags=30, ax=axs[i, 0])
        axs[i, 0].set_title(f'ACF - {title}')
        plot_pacf(series, lags=30, ax=axs[i, 1])
        axs[i, 1].set_title(f'PACF - {title}')
    plt.tight_layout()
    plt.show()

def sarimax_analysis(ts_data, daily_counts):
    exog = ts_data[['temp', 'humidity', 'windspeed', 'weather', 'season']].resample('D').mean().fillna(method='ffill')
    train, test = daily_counts[:-30], daily_counts[-30:]
    exog_train, exog_test = exog.loc[train.index], exog.loc[test.index]

    model = SARIMAX(train, exog=exog_train, order=(1,1,1), seasonal_order=(0,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    forecast = results.get_forecast(steps=30, exog=exog_test)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    mse = ((test - forecast_mean)**2)

    fig, axs = plt.subplots(3, 1, figsize=(14, 10))

    axs[0].plot(test.index, test, label='Actual')
    axs[0].plot(test.index, forecast_mean, label='Forecast', color='red')
    axs[0].fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    axs[0].set_title('SARIMAX Forecast')
    axs[0].legend()

    axs[1].plot(daily_counts.diff().dropna(), color='purple')
    axs[1].set_title('Differenced Series')

    axs[2].plot(mse.index, mse, color='orange', marker='o')
    axs[2].set_title('MSE Loss')

    plt.tight_layout()
    plt.show()

def hybrid_model_analysis(ts_data, daily_counts):
    train, test = daily_counts[:-30], daily_counts[-30:]
    sarimax_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
    residuals = train - sarimax_model.fittedvalues

    exog = ts_data[['temp', 'humidity', 'windspeed', 'weather', 'season']].resample('D').mean().fillna(method='ffill')
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(exog.loc[residuals.index], residuals)

    sarimax_forecast = sarimax_model.get_forecast(steps=30).predicted_mean
    residual_preds = gb.predict(exog.loc[test.index])
    final_forecast = sarimax_forecast + residual_preds

    plt.figure(figsize=(12, 5))
    plt.plot(test.index, test, label='Actual')
    plt.plot(test.index, sarimax_forecast, label='SARIMAX', linestyle='--')
    plt.plot(test.index, final_forecast, label='Hybrid', color='green')
    plt.legend()
    plt.title('Hybrid Forecast')
    plt.show()

    print(f"Hybrid MSE: {mean_squared_error(test, final_forecast):.2f}")
    print(f"SARIMAX MSE: {mean_squared_error(test, sarimax_forecast):.2f}")

def plot_lstm_predictions(model, history, predictions, actual_values):
    train_pred, test_pred = predictions
    y_train_actual, y_test_actual, train_size, test_pred, y_test_actual = actual_values


    plt.figure(figsize=(12, 6))
    plt.plot(y_train_actual, label='Actual Train')
    plt.plot(train_pred, label='Predicted Train')
    plt.plot(range(train_size, train_size + len(test_pred)), y_test_actual, label='Actual Test')
    plt.plot(range(train_size, train_size + len(test_pred)), test_pred, label='Predicted Test')
    plt.legend()
    plt.title('LSTM Model Prediction')
    plt.show()


    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


