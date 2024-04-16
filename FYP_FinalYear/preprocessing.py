from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch

class Preprocessing:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data["DateTime"] = pd.to_datetime(self.data["DateTime"], errors='coerce')
        self.data = self.data.drop(["ID"], axis=1)
        self.scaler = StandardScaler()

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

        num_input_features = X.shape[1]
        print("Number of input features:", num_input_features)

        # Standardize features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

    def run(self):
        self.preprocess_data()

    def preprocess_input(self, input_date):
        # Create an empty DataFrame with the same columns as self.data
        input_df = pd.DataFrame(columns=self.data.columns)

        # Add the input data for each junction
        for junction in range(1, 5):  # Assuming junctions are numbered 1 through 4
            i = junction - 1  # index for input_df
            input_df.loc[i, 'DateTime'] = pd.to_datetime(input_date)
            input_df.loc[i, 'Junction'] = junction

        try:
            input_df['DateTime'] = pd.to_datetime(input_df['DateTime'])
        except Exception as e:
            print("Error:", e)

        # Preprocess the input data as needed
        input_df["Year"] = input_df['DateTime'].dt.year
        input_df["Month"] = input_df['DateTime'].dt.month
        input_df["Date_no"] = input_df['DateTime'].dt.day
        input_df["Day"] = input_df.DateTime.dt.strftime("%A")
        input_df["Hour"] = input_df['DateTime'].dt.hour

        # Perform one-hot encoding for the 'Day' column
        #input_df = pd.get_dummies(input_df, columns=["Day"])

        # Ensure all columns in the original data are present in input_df
        missing_cols = set(self.data.columns) - set(input_df.columns)
        for c in missing_cols:
            input_df[c] = 0

        # Reorder input_df columns to match the original data
        input_df = input_df[self.data.columns]

        # Standardize features using the same scaler as in preprocess_data
        input_array = input_df.drop(['DateTime', 'Vehicles'], axis=1).values
        input_array = self.scaler.transform(input_array)




        return input_array


# Example usage:
data_path = r"C:\Users\rayan\OneDrive\Documents\Year-3\Semester 1\Individual Project\Datasets\traffic.csv"
preprocessing = Preprocessing(data_path)
preprocessing.run()