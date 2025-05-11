from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import AdamW


def build_lstm_model(look_back):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(look_back, 1)))
    model.add(Dropout(0.1))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(GRU(50))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=AdamW(learning_rate=0.001), loss='huber')
    return model
