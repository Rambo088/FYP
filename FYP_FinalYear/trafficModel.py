import tkinter as tk
from tkinter import ttk

import torch
from tkcalendar import Calendar
from gru_model import SymbolicRegressionGRU
from preprocessing import Preprocessing

class TrafficGUI:
    def __init__(self):
        # Paths to your data
        data_path = r"C:\Users\rayan\OneDrive\Documents\Year-3\Semester 1\Individual Project\Datasets\traffic.csv"

        # Initialize preprocessing object and preprocess data
        self.preprocessing = Preprocessing(data_path)
        self.preprocessing.run()

        # Initialize your model (replace 'input_size' with the actual input size of your data)
        input_size = 10  # replace with your actual input size
        self.model = SymbolicRegressionGRU(input_size)

        self.window = tk.Tk()
        self.window.title("Traffic Volume Prediction")

        self.date_label = tk.Label(self.window, text="Enter date (YYYY-MM-DD):")
        self.date_label.pack()

        self.date_entry = tk.Entry(self.window)
        self.date_entry.pack()

        self.predict_button = tk.Button(self.window, text="Predict", command=self.predict)
        self.predict_button.pack()

        self.result_label = tk.Label(self.window, text="")
        self.result_label.pack()

    def predict(self):
        # Get the date from the Entry field
        input_date = self.date_entry.get()

        # Preprocess the input date
        processed_input = self.preprocessing.preprocess_input(input_date)

        # Use the model to make a prediction
        prediction = self.model.predict(processed_input)

        # Update the result label with the prediction
        self.result_label.config(text=f"Predicted traffic volume: {prediction}")

    def run(self):
        self.window.mainloop()

# Initialize the GUI
gui = TrafficGUI()

# Run the GUI
gui.run()
