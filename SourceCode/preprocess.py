import pandas as pd
from tensorflow.keras.optimizers import SGD


class DataProcessor:
    def __init__(self, data):
        self.data = data

    def preprocess_data(data):
        # 'yyyy-mm-dd hh:mm:ss'
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        # Extracting features from datetime
        data['hour'] = data['DateTime'].dt.hour
        data['day'] = data['DateTime'].dt.day
        data['weekday'] = data['DateTime'].dt.weekday
        # Dropping ID column
        data.drop(columns=['ID'], inplace=True)

        return data


data_path = r"C:\Users\rayan\OneDrive\Documents\Year-3\Semester 1\Individual Project\Datasets\traffic.csv"
data = pd.read_csv(data_path)
