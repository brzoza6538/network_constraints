import matplotlib.pyplot as plt
from typing import Literal
import numpy as np

from algorithms.differential import DifferentialEvolutionAlgorithm
from algorithms.genetic import GeneticAlgorithm
from data_reader import parse_sndlib_file


def run_genetic_algorithm(
    data,
    n_generations,
    cross_aggregating,
    population_size,
    tournament_size,
    survivors,
    severity_of_mutation,
    mutation_aggregation_chance,
    normal_mutation_chance,
    switch_mutation_chance,
    num_of_init_chunks,
) -> list[int]:
    
    """
    Runs the genetic algorithm.
    :param file_path: Path to the file with the SNDlib data.
    :param n_generations: Number of generations to run the algorithm.
    """

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
    for i in range(n_generations):
        generation = algorithm.run_generation()
        costs = [algorithm.evaluate_cost(gene) for gene in generation]
        generation_min = min(costs)
        result.append(generation_min)

        # generation_max = max(costs)
        # generation_avg = sum(costs)/len(costs)
        # print(f"Generation {i}:")
        # print(f"\tMin: {generation_min}, Max: {generation_max}, Avg: {generation_avg}")

    return result


def test_genetic_algorithm(
    data,
    n_runs: int = 5,
    n_generations: int = 150,
    cross_aggregating: bool = True,
    population_size: int = 100,
    tournament_size: int = 2,
    survivors: int = 15,
    severity_of_mutation: float = 0.8,
    mutation_chance: float = 0.8,
    mutation_type: Literal["normal", "aggregation", "switch"] = "normal",
    num_of_init_chunks: int = 80,
) -> tuple:
    """
    Runs the genetic algorithm multiple times.
    Returns the median and standard deviation of the results.
    """

    normal_mutation_chance = mutation_chance if mutation_type == "normal" else 0.0
    mutation_aggregation_chance = mutation_chance if mutation_type == "aggregation" else 0.0
    switch_mutation_chance = mutation_chance if mutation_type == "switch" else 0.0

    all_results = []
    for _ in range(n_runs):
        result = run_genetic_algorithm(
            data,
            n_generations=n_generations,
            cross_aggregating=cross_aggregating,
            population_size=population_size,
            severity_of_mutation=severity_of_mutation,
            normal_mutation_chance=normal_mutation_chance,
            mutation_aggregation_chance=mutation_aggregation_chance,
            switch_mutation_chance=switch_mutation_chance,
            tournament_size=tournament_size,
            survivors=survivors,
            num_of_init_chunks=num_of_init_chunks,
        )
        all_results.append(result)

    all_results = np.array(all_results)
    median_results = np.median(all_results, axis=0)
    std_results = np.std(all_results, axis=0)

    return median_results, std_results


def plot_results(title, values, medians, stds):
    plt.figure(figsize=(10, 5))

    for i, value in enumerate(values):
        plt.plot(medians[i], label=value)
        plt.fill_between(
            range(len(medians[i])),
            medians[i] - stds[i],
            medians[i] + stds[i],
            alpha=0.2,
        )

    plt.xlabel("Generation")
    plt.ylabel("Cost")
    plt.yscale("log")
    plt.title("Results")
    plt.legend(title=title)
    plt.show()


def run_differential_algorithm(
    data,
    n_generations,
    population_size,
    diff_F,
    diff_CR,
    parental_tournament_size,
    survivors_amount,
    smoothing_mutation_chance,
    num_of_init_chunks,
) -> list[int]:
    """
    Runs the genetic algorithm.
    :param file_path: Path to the file with the SNDlib data.
    :param n_generations: Number of generations to run the algorithm.
    """

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
        num_of_init_chunks=num_of_init_chunks
    )

    result = []
    algorithm.generate_genes()

    for i in range(n_generations):
        generation = algorithm.run_generation()
        costs = [algorithm.evaluate_cost(gene) for gene in generation]
        generation_min = min(costs)
        result.append(generation_min)

        # TODO: Remove this
        # generation_max = max(costs)
        # generation_avg = sum(costs) / len(costs)
        # print(f"Generation {i}:")
        # print(f"\tMin: {generation_min}, Max: {generation_max}, Avg: {generation_avg}")

    return result


def test_differential_algorithm(
    data,
    n_runs: int = 2,
    n_generations: int = 100,
    population_size: int = 1000,
    diff_F: float = 1,
    diff_CR: float = 0.8,
    parental_tournament_size: int = 1,
    survivors_amount: int = 50,
    smoothing_mutation_chance: float = 0.08,
    num_of_init_chunks: int = 50,
) -> tuple:
    """
    Runs the differential algorithm multiple times.
    Returns the median and standard deviation of the results.
    """

    all_results = []
    for _ in range(n_runs):
        result = run_differential_algorithm(
            data,
            n_generations=n_generations,
            population_size=population_size,
            diff_F=diff_F,
            diff_CR=diff_CR,
            parental_tournament_size=parental_tournament_size,
            survivors_amount=survivors_amount,
            smoothing_mutation_chance=smoothing_mutation_chance,
            num_of_init_chunks=num_of_init_chunks,
        )
        all_results.append(result)

    all_results = np.array(all_results)
    median_results = np.median(all_results, axis=0)
    std_results = np.std(all_results, axis=0)

    return median_results, std_results


if __name__ == "__main__":
    with open("data.txt", "r") as file:
        file_content = file.read()
    data = parse_sndlib_file(file_content)

    result = test_genetic_algorithm(data, cross_aggregating=True, mutation_type="normal", num_of_init_chunks=50, mutation_chance=0.8)

    print(f"Result: {result}")
