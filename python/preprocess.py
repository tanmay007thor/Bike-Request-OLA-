import pandas as pd

def load_and_process_data(path='../data/ola_updated.csv'):
    data = pd.read_csv(path)
    data['time'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
    ts_data = data.set_index('time')
    daily_counts = ts_data['count'].resample('D').sum().dropna()
    return data, ts_data, daily_counts
