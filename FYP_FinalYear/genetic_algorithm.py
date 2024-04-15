import random

class Genetic_algorithm:
    def __init__(self, population_size, num_generations):
        self.population_size = population_size
        self.num_generations = num_generations

    def initialize_population(self):
        # Generate an initial population of individuals
        population = []
        for _ in range(self.population_size):
            # Generate random hyperparameters for GRU model
            hyperparameters = {
                "hidden_units": random.randint(32, 256),
                "learning_rate": random.uniform(0.0001, 0.01),
                "dropout_rate": random.uniform(0.0, 0.5),
                # Add more hyperparameters as needed
            }
            population.append(hyperparameters)
        return population

    def evaluate_population(self, population):
        # Train and evaluate each individual in the population
        scores = []
        for hyperparameters in population:
            # Train and evaluate GRU model using hyperparameters
            score = train_and_evaluate_GRU_model(hyperparameters)
            scores.append((hyperparameters, score))
        return scores

    def selection(self, scores):
        # Select individuals from the population based on their scores
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        selected_individuals = [score[0] for score in sorted_scores[:self.population_size // 2]]
        return selected_individuals

    def crossover(self, selected_individuals):
        # Create new individuals (offspring) by combining selected parents
        offspring = []
        for i in range(0, len(selected_individuals), 2):
            parent1 = selected_individuals[i]
            parent2 = selected_individuals[i + 1]
            # Implement crossover logic (e.g., single-point crossover)
            crossover_point = random.randint(1, len(parent1) - 1)
            offspring1 = {**parent1, **parent2}
            offspring2 = {**parent2, **parent1}
            offspring.append(offspring1)
            offspring.append(offspring2)
        return offspring

    def mutation(self, offspring):
        # Introduce random changes to maintain genetic diversity
        mutated_offspring = []
        for individual in offspring:
            mutated_individual = individual.copy()
            # Implement mutation logic (e.g., randomly modify hyperparameters)
            for key in mutated_individual:
                if random.random() < mutation_rate:
                    if key == "hidden_units":
                        mutated_individual[key] = random.randint(32, 256)
                    elif key == "learning_rate":
                        mutated_individual[key] = random.uniform(0.0001, 0.01)
                    elif key == "dropout_rate":
                        mutated_individual[key] = random.uniform(0.0, 0.5)
                    # Add more mutation logic as needed
            mutated_offspring.append(mutated_individual)
        return mutated_offspring

    def evaluate_best_individual(self, best_individual):
        # Evaluate the best individual in the final population
        # Train and test GRU model using the best hyperparameters
        train_and_test_GRU_model(best_individual)

    def run_genetic_algorithm(self):
        # Initialize population
        population = self.initialize_population()

        for generation in range(self.num_generations):
            # Evaluate population
            scores = self.evaluate_population(population)

            # Select individuals for reproduction
            selected_individuals = self.selection(scores)

            # Create offspring through crossover
            offspring = self.crossover(selected_individuals)

            # Mutate offspring
            mutated_offspring = self.mutation(offspring)

            # Combine current population and mutated offspring
            population = selected_individuals + mutated_offspring

        # Evaluate and select the best individual
        best_individual = max(scores, key=lambda x: x[1])[0]

        # Evaluate the best individual
        self.evaluate_best_individual(best_individual)
