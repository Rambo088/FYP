import pandas as pd
from tensorflow.keras.models import load_model

def preprocess_data(data):
    # Assuming 'DateTime' column is in the format of 'yyyy-mm-dd hh:mm:ss'
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    # Extracting features from datetime
    data['hour'] = data['DateTime'].dt.hour
    data['day'] = data['DateTime'].dt.day
    data['weekday'] = data['DateTime'].dt.weekday
    # Dropping ID column
    data.drop(columns=['ID'], inplace=True)
    return data

# Load your dataset
data_path = r"C:\Users\rayan\OneDrive\Documents\Year-3\Semester 1\Individual Project\Datasets\traffic.csv"
data = pd.read_csv(data_path)
# Preprocess data
processed_data = preprocess_data(data)
# Split data into features and target
X = processed_data[['hour', 'day', 'weekday']].values  # Assuming 'weekday' is also included
# Assuming 'Junction' column represents the target variable
y = processed_data['Junction'].values

# Load the trained model
trained_model = load_model('trained_model.h5')

# Function to predict traffic
def predict_traffic(X):
    # Make prediction using the trained model
    prediction = trained_model.predict(X)
    return prediction

# Predict traffic for the dataset
predicted_traffic = predict_traffic(X)
print(predicted_traffic)
