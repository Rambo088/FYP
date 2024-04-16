import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense
from keras import callbacks
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.layers import BatchNormalization

initial_learning_rate = 0.1
decay_steps = 1000
decay_rate = 0.9
lr_schedule = ExponentialDecay(
    initial_learning_rate, decay_steps, decay_rate, staircase=True
)


class SymbolicRegressionGRU(nn.Module):
    def __init__(self, input_size):
        super(SymbolicRegressionGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        # No need to add an extra dimension here as it's already handled in preprocess_input
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, x):
        # Convert numpy array to PyTorch tensor if necessary
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Add an extra dimension for GRU input if necessary
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        # Ensure the model is in evaluation mode
        self.eval()

        # Perform the forward pass and return the prediction
        with torch.no_grad():
            prediction = self.forward(x)
        return prediction


def GRU_model(X_Train, y_Train, X_Test, y_Test):
    early_stopping = callbacks.EarlyStopping(min_delta=0.001, patience=10, restore_best_weights=True)

    # The GRU model
    model = Sequential()
    model.add(GRU(units=150, return_sequences=True, input_shape=(X_Train.shape[1], 1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(units=150, return_sequences=True, input_shape=(X_Train.shape[1], 1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=True, input_shape=(X_Train.shape[1], 1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=True, input_shape=(X_Train.shape[1], 1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, input_shape=(X_Train.shape[1], 1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compiling the model
    model.compile(optimizer=SGD(learning_rate=lr_schedule, momentum=0.9), loss='mean_squared_error')
    model.fit(X_Train, y_Train, validation_data=(X_Test, y_Test), epochs=50, batch_size=120, callbacks=[early_stopping])
    pred_GRU = model.predict(X_Test)
    return pred_GRU
