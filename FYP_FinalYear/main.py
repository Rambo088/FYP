import pandas as pd
from preprocess import DataProcessor
from train_model import GRU_model

def main():
    # Load your dataset
    data_path = r"C:\Users\rayan\OneDrive\Documents\Year-3\Semester 1\Individual Project\Datasets\traffic.csv"
    data = pd.read_csv(data_path)

    # Instantiate DataProcessor object
    processor = DataProcessor(data)
    processed_data = processor.preprocess_data()

    # Preprocess the data
    processed_data = processor.preprocess_data()

    # Train your GRU model
    X = processed_data[['hour', 'day', 'weekday']].values
    y = processed_data['Junction'].values
    GRU_model(X, y)

if __name__ == "__main__":
    main()
