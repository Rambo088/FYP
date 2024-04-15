import tkinter as tk
from tkinter import messagebox
from genetic_algorithm import Genetic_algorithm
from gru_model import SymbolicRegressionGRU
from preprocessing import Preprocessing
from trafficModel import TrafficGUI

def main():
    # Get input data
    input_data = date_entry.get()
    data_path = r"C:\Users\rayan\OneDrive\Documents\Year-3\Semester 1\Individual Project\Datasets\traffic.csv"
    # Preprocess input data
    preprocessing = Preprocessing(data_path)
    processed_input = preprocessing.preprocess_input(input_data)
    print(input_data)


    # Initialize and run genetic algorithm
    geneticAlgorithm = Genetic_algorithm(population_size=100, num_generations=50)
    best_individual = Genetic_algorithm.optimize()

    # Load trained model
    model = TrafficGUI(SymbolicRegressionGRU, data_path)
    prediction = model.predict('2022-01-01')
    print(f"The predicted traffic volume for 2022-01-01 is: {prediction}")
    # Use the best individual to set model parameters or hyperparameters
    # model_parameters = ...

    # Make predictions using the model
    predictions = model(processed_input)

    # Postprocess predictions if necessary
    processed_predictions = predictions  # No postprocessing for now

    # Show the prediction
    messagebox.showinfo("Prediction", f"The predicted traffic volume is: {processed_predictions}")

# Create a Tkinter window
root = tk.Tk()

# Create a label and entry for the date input
date_label = tk.Label(root, text="Enter a date:")
date_label.pack()
date_entry = tk.Entry(root)
date_entry.pack()

# Create a button that will call the main function when clicked
predict_button = tk.Button(root, text="Predict", command=main)
predict_button.pack()

# Run the Tkinter event loop
root.mainloop()
