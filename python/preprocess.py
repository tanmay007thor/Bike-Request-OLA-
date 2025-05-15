import pandas as pd
import numpy as np

def load_and_process_data(path='../data/ola_updated.csv'):
    data = pd.read_csv(path)
    data['time'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
    data['day_of_week'] = data['time'].dt.weekday.apply(lambda x: (x + 1) % 7)
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x in [0, 6] else 0)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['hour_bin'] = pd.cut(data['hour'], bins=[-1, 6, 12, 17, 21, 24], labels=[0,1,2,3,4]).astype(int)
    data['temp_humidity'] = data['temp'] * data['humidity']
    data['wind_temp_ratio'] = data['windspeed'] / (data['temp'] + 1e-3)
    data['lag_1'] = data['count'].shift(1)
    data['lag_2'] = data['count'].shift(2)
    data['roll_mean_3'] = data['count'].rolling(3).mean()
    data['roll_std_3'] = data['count'].rolling(3).std()
    data.dropna(inplace=True)
    ts_data = data.set_index('time')
    daily_counts = ts_data['count'].resample('D').sum().dropna()
    features = data[['season','weather','temp','humidity','windspeed',
                     'hour_sin','hour_cos','day_of_week','is_weekend',
                     'month_sin','month_cos','hour_bin',
                     'temp_humidity','wind_temp_ratio',
                     'lag_1','lag_2','roll_mean_3','roll_std_3']]
    features.index = data['time']
    return data, ts_data, daily_counts, features
