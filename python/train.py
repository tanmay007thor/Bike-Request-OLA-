import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model import build_lstm_model

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def train_lstm_model(ts_data, look_back):
    daily_counts = ts_data['count'].resample('D').sum().dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(daily_counts.values.reshape(-1, 1))

    X, y = create_sequences(scaled, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = build_lstm_model(look_back)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    train_pred = scaler.inverse_transform(model.predict(X_train))
    test_pred = scaler.inverse_transform(model.predict(X_test))
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(12, 6))
    plt.plot(y_train_actual, label='Actual Train')
    plt.plot(train_pred, label='Predicted Train')
    plt.plot(range(train_size, train_size + len(test_pred)), y_test_actual, label='Actual Test')
    plt.plot(range(train_size, train_size + len(test_pred)), test_pred, label='Predicted Test')
    plt.legend()
    plt.title('LSTM Model Prediction')
    plt.show()

    return model, history, (train_pred, test_pred)
