import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model import build_lstm_model

def create_sequences(X_data, y_data, look_back):
    X, y = [], []
    for i in range(len(X_data) - look_back):
        X.append(X_data[i:i + look_back])
        y.append(y_data[i + look_back])
    return np.array(X), np.array(y)
def train_lstm_model(ts_data, features, look_back):
    daily_counts = ts_data['count'].resample('D').sum().dropna()
    daily_features = features.resample('D').mean().loc[daily_counts.index]

    combined = daily_features.copy()
    combined['count'] = daily_counts

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(combined.drop(columns='count'))
    y_scaled = scaler_y.fit_transform(combined['count'].values.reshape(-1, 1))

    X, y = create_sequences(X_scaled, y_scaled, look_back)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = build_lstm_model(look_back, X.shape[2])  
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    model.save('saved_lstm_model.h5') 
    train_pred = scaler_y.inverse_transform(model.predict(X_train))
    test_pred = scaler_y.inverse_transform(model.predict(X_test))
    y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    return model, history, (train_pred, test_pred), (y_train_actual, train_pred, train_size, test_pred, y_test_actual)
