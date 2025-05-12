import pandas as pd
import numpy as np

def load_and_process_data(path='../data/ola_updated.csv'):
    data = pd.read_csv(path)
    data['time'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    ts_data = data.set_index('time')
    daily_counts = ts_data['count'].resample('D').sum().dropna()
    features = data[['season', 'weather', 'temp', 'humidity', 'windspeed', 'hour_sin', 'hour_cos']]
    features.index = ts_data.index 
    return data, ts_data, daily_counts, features
