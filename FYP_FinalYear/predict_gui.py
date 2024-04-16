import tkinter as tk
from tkinter import ttk
from datetime import datetime
import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained model
trained_model = load_model('trained_model.h5')

# Function to predict traffic based on user input
def predict_traffic():
    # Get user input from GUI
    date_str = date_entry.get()
    time_str = time_entry.get()

    # Convert user input to datetime object
    datetime_obj = datetime.strptime(date_str + ' ' + time_str, '%Y-%m-%d %H:%M:%S')

    # Extract features from datetime object
    hour = datetime_obj.hour
    day = datetime_obj.day
    weekday = datetime_obj.weekday()

    # Prepare input data for prediction
    input_data = pd.DataFrame({'hour': [hour], 'day': [day], 'weekday': [weekday]})
    input_data = input_data.values

    # Make prediction using the trained model
    prediction = trained_model.predict(input_data)

    # Display prediction
    prediction_label.config(text=f'Predicted vehicles: {prediction[0][0]}')

# Create GUI window
window = tk.Tk()
window.title("Traffic Prediction")

# Date input
date_label = ttk.Label(window, text="Date (YYYY-MM-DD):")
date_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
date_entry = ttk.Entry(window, width=20)
date_entry.grid(row=0, column=1, padx=10, pady=5)

# Time input
time_label = ttk.Label(window, text="Time (HH:MM:SS):")
time_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
time_entry = ttk.Entry(window, width=20)
time_entry.grid(row=1, column=1, padx=10, pady=5)

# Button to trigger prediction
predict_button = ttk.Button(window, text="Predict", command=predict_traffic)
predict_button.grid(row=2, column=0, columnspan=2, pady=10)

# Label to display prediction result
prediction_label = ttk.Label(window, text="")
prediction_label.grid(row=3, column=0, columnspan=2)

# Run the GUI
window.mainloop()
