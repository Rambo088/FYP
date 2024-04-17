import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from preprocess import DataProcessor


def GRU_model(X_Train, y_Train):

    # The GRU model
    model = Sequential()
    model.add(GRU(units=150, return_sequences=True, input_shape=(X_Train.shape[1], 1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(units=150, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compiling the model
    lr_schedule = 0.01  # You need to define your learning rate schedule here
    model.compile(optimizer=SGD(learning_rate=lr_schedule, momentum=0.9), loss='mean_squared_error')
    model.fit(X_Train, y_Train, epochs=50, batch_size=120)

    # Save the trained model
    model.save('trained_model.h5')


# Load your dataset
data_path = r"C:\Users\rayan\OneDrive\Documents\Year-3\Semester 1\Individual Project\Datasets\traffic.csv"
data = pd.read_csv(data_path)
# Preprocess data
processed_data = DataProcessor.preprocess_data(data)
# Split data into features and target
X = processed_data[['hour', 'day', 'weekday']].values  # Assuming 'weekday' is also included
# Assuming 'Junction' column represents the target variable
y = processed_data['Junction'].values
# Train your GRU model and save it
GRU_model(X, y)