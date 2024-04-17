import numpy as np
from sklearn.model_selection import train_test_split
from train_model import GRU_model

class GeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            # Randomly generate hyperparameters
            # Example: {'units': np.random.randint(50, 200), 'dropout': np.random.uniform(0.1, 0.5)}
            hyperparameters = {'units': np.random.randint(50, 200), 'dropout': np.random.uniform(0.1, 0.5)}
            population.append(hyperparameters)
        return population

    def fitness(self, hyperparameters, X_train, y_train):
        # Evaluate fitness by training the GRU model with given hyperparameters
        model = GRU_model(**hyperparameters)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        model.train_model(X_train, y_train)
        # Calculate fitness based on validation loss or any other metric you prefer
        val_loss = model.evaluate(X_val, y_val)
        return 1 / (1 + val_loss)  # Maximizing fitness

    def selection(self, population, X_train, y_train):
        # Select top performers based on fitness
        sorted_population = sorted(population, key=lambda x: self.fitness(x, X_train, y_train), reverse=True)
        return sorted_population[:int(0.2 * self.population_size)]  # Selecting top 20%

    def crossover(self, population):
        # Perform crossover to generate new offspring
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = np.random.choice(population, 2, replace=False)
            child = {key: (parent1[key] + parent2[key]) / 2 for key in parent1.keys()}
            new_population.append(child)
        return new_population

    def mutate(self, population):
        # Mutate the population with a certain probability
        for i in range(len(population)):
            if np.random.rand() < self.mutation_rate:
                population[i]['units'] = np.random.randint(50, 200)
                population[i]['dropout'] = np.random.uniform(0.1, 0.5)
        return population

    def optimize_hyperparameters(self, X_train, y_train):
        population = self.initialize_population()
        for _ in range(self.generations):
            population = self.selection(population, X_train, y_train)
            population = self.crossover(population)
            population = self.mutate(population)
        best_hyperparameters = population[0]
        return best_hyperparameters

# Example usage
data_path = r"C:\Users\rayan\OneDrive\Documents\Year-3\Semester 1\Individual Project\Datasets\traffic.csv"
traffic_model = GRU_model(data_path)
X_train, y_train = traffic_model.preprocess_data()

genetic_algorithm = GeneticAlgorithm(population_size=50, generations=10, mutation_rate=0.1)
best_hyperparameters = genetic_algorithm.optimize_hyperparameters(X_train, y_train)
print("Best hyperparameters found:", best_hyperparameters)
