import matplotlib.pyplot as plt
from typing import Literal
import numpy as np

from algorithms.differential import DifferentialEvolutionAlgorithm
from algorithms.genetic import GeneticAlgorithm
from data_reader import parse_sndlib_file


def test_genetic_algorithm(
    data,
    n_runs: int = 10,
    n_generations: int = 150,
    cross_aggregating: bool = True,
    population_size: int = 150,
    tournament_size: int = 2,
    survivors: int = 10,
    severity_of_mutation: float = 0.8,
    mutation_chance: float = 0.5,
    mutation_type: str = "normal",
    num_of_init_chunks: int = 50,
) -> tuple:
    """
    Runs the genetic algorithm multiple times and computes statistics.
    Returns the median and standard deviation of the results,
    along with a list of cost function counter values for each generation.
    """

    normal_mutation_chance = mutation_chance if mutation_type == "normal" else 0.0
    mutation_aggregation_chance = mutation_chance if mutation_type == "aggregation" else 0.0
    switch_mutation_chance = mutation_chance if mutation_type == "switch" else 0.0

    all_results = []
    cost_function_counters = []
    for _ in range(n_runs):
        # Initialize and run the genetic algorithm
        nodes, links, demands, admissible_paths = data
        algorithm = GeneticAlgorithm(
            nodes,
            links,
            demands,
            admissible_paths,
            cross_aggregating=cross_aggregating,
            population_size=population_size,
            severity_of_mutation=severity_of_mutation,
            mutation_aggregation_chance=mutation_aggregation_chance,
            normal_mutation_chance=normal_mutation_chance,
            switch_mutation_chance=switch_mutation_chance,
            tournament_size=tournament_size,
            survivors=survivors,
            num_of_init_chunks=num_of_init_chunks,
        )
        algorithm.generate_genes()
        
        result = []
        cost_counters = []
        for _ in range(n_generations):
            generation = algorithm.run_generation()
            costs = [algorithm.evaluate_cost(gene) for gene in generation]
            result.append(min(costs))
            cost_counters.append(algorithm.cost_function_counter)
        
        all_results.append(result)
        cost_function_counters.append(cost_counters)

    all_results = np.array(all_results)
    cost_counters = np.median(cost_function_counters, axis=0)

    return all_results, cost_counters



def test_differential_algorithm(
    data,
    n_runs: int = 5,
    n_generations: int = 100,
    population_size: int = 1000,
    diff_F: float = 1,
    diff_CR: float = 0.8,
    parental_tournament_size: int = 1,
    survivors_amount: int = 50,
    smoothing_mutation_chance: float = 0.05,
    num_of_init_chunks: int = 1,
) -> tuple:
    """
    Runs the differential evolution algorithm multiple times and computes statistics.
    Returns the median and standard deviation of the results.
    """
    all_results = []
    cost_function_counters = []

    for _ in range(n_runs):
        nodes, links, demands, admissible_paths = data
        algorithm = DifferentialEvolutionAlgorithm(
            nodes,
            links,
            demands,
            admissible_paths,
            population_size=population_size,
            diff_F=diff_F,
            diff_CR=diff_CR,
            parental_tournament_size=parental_tournament_size,
            survivors_amount=survivors_amount,
            smoothing_mutation_chance=smoothing_mutation_chance,
            num_of_init_chunks=num_of_init_chunks,
        )

        algorithm.generate_genes()
        result = []
        cost_counters = []

        for _ in range(n_generations):
            generation = algorithm.run_generation()
            costs = [algorithm.evaluate_cost(gene) for gene in generation]
            result.append(min(costs))
            cost_counters.append(algorithm.cost_function_counter)

        all_results.append(result)
        cost_function_counters.append(cost_counters)
    
    all_results = np.array(all_results)
    cost_counters = np.median(cost_function_counters, axis=0)
    return all_results, cost_counters


if __name__ == "__main__":
    with open("data.txt", "r") as file:
        file_content = file.read()
    data = parse_sndlib_file(file_content)

    result = test_genetic_algorithm(data, cross_aggregating=True, mutation_type="normal", num_of_init_chunks=50, mutation_chance=0.8)

    print(f"Result: {result}")



def plot_results(title, values, medians, stds, costs_counter):
    plt.figure(figsize=(10, 5))

    for i, value in enumerate(values):
        plt.plot(costs_counter[i], medians[i], label=value)
        plt.fill_between(
            costs_counter[i],
            medians[i] - stds[i],
            medians[i] + stds[i],
            alpha=0.2,
        )

    plt.xlabel("costs counted")
    plt.ylabel("min cost found")
    plt.yscale("log")
    plt.title("Results")
    plt.legend(title=title)
    plt.show()