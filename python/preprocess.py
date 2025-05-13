import pandas as pd
import numpy as np

def load_and_process_data(path='../data/ola_updated.csv'):
    data = pd.read_csv(path)
    data['time'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
    data['day_of_week'] = pd.to_datetime(data['time']).dt.weekday.apply(lambda x: (x + 1) % 7)
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x in [0, 6] else 0)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    ts_data = data.set_index('time')
    daily_counts = ts_data['count'].resample('D').sum().dropna()
    features = data[['season', 'weather', 'temp', 'humidity', 'windspeed', 'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend']]
    features.index = ts_data.index
    return data, ts_data, daily_counts, features
