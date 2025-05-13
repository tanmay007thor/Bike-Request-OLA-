from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber

def build_lstm_model(look_back, num_features):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(look_back, num_features)))
    model.add(BatchNormalization())
    model.add(LSTM(100, return_sequences=True))
    model.add(BatchNormalization())
    model.add(GRU(100, return_sequences=True, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(GRU(50, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=Huber())
    return model
