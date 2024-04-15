from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

class Preprocessing:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data["DateTime"] = pd.to_datetime(self.data["DateTime"], errors='coerce')
        self.data = self.data.drop(["ID"], axis=1)
        self.scaler = StandardScaler()

    def plot_time_series(self):
        # Plotting Time Series
        colors = ["#FF0000", "#0006FF", "#1DFF00", "#FAFF00"]
        plt.figure(figsize=(25, 8), facecolor="#99ccff")
        time_series_plot = sns.lineplot(x=self.data['DateTime'], y="Vehicles", data=self.data, hue="Junction", palette=colors)
        time_series_plot.set_title("Each Junction's Traffic Over The Years")
        time_series_plot.set_xlabel("Date")
        time_series_plot.set_ylabel("Number of Vehicles")
        plt.show()

    def preprocess_data(self):
        self.data["Year"] = self.data['DateTime'].dt.year
        self.data["Month"] = self.data['DateTime'].dt.month
        self.data["Date_no"] = self.data['DateTime'].dt.day
        self.data["Day"] = self.data.DateTime.dt.strftime("%A")
        self.data["Hour"] = self.data['DateTime'].dt.hour

        # Perform one-hot encoding for categorical variables
        categorical_cols = ["Day"]
        self.data = pd.get_dummies(self.data, columns=categorical_cols)

        X = self.data.drop(['DateTime', 'Vehicles'], axis=1).values
        y = self.data['Vehicles'].values.reshape(-1, 1)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Convert data to PyTorch tensors
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = y_test

    def train_simple_regression_model(self):
        # Define a simple neural network for regression
        class SimpleRegressionModel(nn.Module):
            def __init__(self, input_size):
                super(SimpleRegressionModel, self).__init__()
                self.linear1 = nn.Linear(input_size, 64)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(64, 1)

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        # Instantiate the model
        input_size = self.X_train_tensor.shape[1]
        model = SimpleRegressionModel(input_size)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training the model
        num_epochs = 1000

        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(self.X_train_tensor)
            loss = criterion(outputs, self.y_train_tensor)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Make predictions on the test set
        with torch.no_grad():
            predictions = model(self.X_test_tensor).numpy()

        # Calculate model performance
        self.model_performance = 100 - mean_squared_error(self.y_test, predictions) * 100 / np.var(self.y_test)
        print(f'Model Performance on Training Set: {self.model_performance:.2f}%')

    def train_GRU_model(self):
        # Define a GRU model for symbolic regression
        class SymbolicRegressionGRU(nn.Module):
            def __init__(self, input_size):
                super(SymbolicRegressionGRU, self).__init__()
                self.gru = nn.GRU(input_size, hidden_size=50, num_layers=2, batch_first=True)
                self.fc = nn.Linear(50, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                out = self.fc(out[:, -1, :])
                return out

        # Instantiate the model
        input_size = self.X_train_tensor.shape[1]
        model = SymbolicRegressionGRU(input_size)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training the model
        num_epochs = 1000

        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(self.X_train_tensor.unsqueeze(1))
            loss = criterion(outputs, self.y_train_tensor)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Make predictions on the test set
        with torch.no_grad():
            predictions = model(self.X_test_tensor.unsqueeze(1)).numpy()

        # Calculate model performance
        self.model_performance_GRU = 100 - mean_squared_error(self.y_test, predictions) * 100 / np.var(self.y_test)
        print(f'Model Performance on Testing Set: {self.model_performance_GRU:.2f}%')

    def run(self):
        self.preprocess_data()
        self.train_simple_regression_model()
        self.train_GRU_model()

    def preprocess_input(self, input_date):
        # Create an empty DataFrame with the same columns as self.data
        input_df = pd.DataFrame(columns=self.data.columns)

        # Add the input data for each junction
        for junction in range(1, 5):  # Assuming junctions are numbered 1 through 4
            i = junction - 1  # index for input_df
            input_df.loc[i, 'DateTime'] = pd.to_datetime(input_date)
            input_df.loc[i, 'Junction'] = junction

        print("Type of input_df['DateTime']:", type(input_df['DateTime']))
        print("Content of input_df['DateTime']:", input_df['DateTime'])

        try:
            input_df['DateTime'] = pd.to_datetime(input_df['DateTime'])
        except Exception as e:
            print("Error:", e)

        print("Type of input_df['DateTime']:", type(input_df['DateTime']))
        print("Content of input_df['DateTime']:", input_df['DateTime'])

        # Preprocess the input data as needed
        input_df["Year"] = input_df['DateTime'].dt.year
        input_df["Month"] = input_df['DateTime'].dt.month
        input_df["Date_no"] = input_df['DateTime'].dt.day
        input_df["Day"] = input_df.DateTime.dt.strftime("%A")
        input_df["Hour"] = input_df['DateTime'].dt.hour

        # Perform one-hot encoding for categorical variables
        input_df = pd.get_dummies(input_df, columns=["Day"])

        # Ensure all columns in the original data are present in input_df
        missing_cols = set(self.data.columns) - set(input_df.columns)
        for c in missing_cols:
            input_df[c] = 0

        # Reorder input_df columns to match the original data
        input_df = input_df[self.data.columns]

        # Standardize features using the same scaler as in preprocess_data
        input_array = input_df.drop('DateTime', axis=1).values
        input_array = self.scaler.transform(input_array)

        return input_array


# Example usage:
data_path = r"C:\Users\rayan\OneDrive\Documents\Year-3\Semester 1\Individual Project\Datasets\traffic.csv"
preprocessing = Preprocessing(data_path)
preprocessing.run()
