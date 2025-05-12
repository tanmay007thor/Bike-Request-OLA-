from preprocess import load_and_process_data
from analysis import visualize_time_series, plot_acf_pacf, sarimax_analysis, hybrid_model_analysis , plot_lstm_predictions
from model import build_lstm_model
from train import train_lstm_model


data, ts_data, daily_counts , features = load_and_process_data()
visualize_time_series(ts_data)
plot_acf_pacf(daily_counts)
sarimax_analysis(ts_data, daily_counts)
hybrid_model_analysis(ts_data, daily_counts)
model, history, predictions , actual_values = train_lstm_model(ts_data, features , look_back=30)
plot_lstm_predictions(model, history, predictions, actual_values)