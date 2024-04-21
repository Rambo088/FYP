import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from preprocess import DataProcessor
from sklearn.model_selection import train_test_split

class TrafficGRUModel:
    def __init__(self, units=150, dropout=0.2, learning_rate=0.01, momentum=0.9, epochs=50, batch_size=120):
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def preprocess_data(self, data_path):
        data = pd.read_csv(data_path)
        processed_data = DataProcessor.preprocess_data(data)
        X = processed_data[['hour', 'day', 'weekday']].values
        y = processed_data['Junction'].values
        return X, y

    def train_model(self, X_train, y_train):
        self.model = Sequential([
            GRU(units=self.units, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'),
            Dropout(self.dropout),
            GRU(units=self.units, return_sequences=True, activation='tanh'),
            Dropout(self.dropout),
            GRU(units=int(self.units/3), return_sequences=True, activation='tanh'),
            Dropout(self.dropout),
            GRU(units=int(self.units/3), return_sequences=True, activation='tanh'),
            Dropout(self.dropout),
            GRU(units=int(self.units/3), activation='tanh'),
            Dropout(self.dropout),
            Dense(units=1)
        ])
        optimizer = SGD(learning_rate=self.learning_rate, momentum=self.momentum)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

        # Save the trained model
        self.model.save('trained_model.h5')

    def evaluate(self, X_test, y_test):
        loss = self.model.evaluate(X_test, y_test)
        return loss


# Load your dataset
data_path = r"C:\Users\rayan\OneDrive\Documents\Year-3\Semester 1\Individual Project\Datasets\traffic.csv"
traffic_model = TrafficGRUModel()
X, y = traffic_model.preprocess_data(data_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
traffic_model.train_model(X_train, y_train)
test_loss = traffic_model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
