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
    )
    algorithm.generate_genes()

    result = []
    for i in range(n_generations):
        generation = algorithm.run_generation()
        costs = [algorithm.evaluate_cost(gene) for gene in generation]
        generation_min = min(costs)
        result.append(generation_min)

        # TODO: Remove this
        # generation_max = max(costs)
        # generation_avg = sum(costs)/len(costs)
        # print(f"Generation {i}:")
        # print(f"\tMin: {generation_min}, Max: {generation_max}, Avg: {generation_avg}")

    return result


def test_genetic_algorithm(
    data,
    n_runs: int = 25,
    n_generations: int = 100,
    cross_aggregating: bool = True,
    population_size: int = 150,
    tournament_size: int = 2,
    survivors: int = 10,
    severity_of_mutation: float = 0.8,
    mutation_chance: float = 0.9,
    mutation_type: Literal["normal", "aggregation", "switch"] = "normal",
) -> tuple:
    """
    Runs the genetic algorithm multiple times.
    Returns the median and standard deviation of the results.
    """

    normal_mutation_chance = mutation_chance if mutation_type == "normal" else 0.0
    mutation_aggregation_chance = (
        mutation_chance if mutation_type == "aggregation" else 0.0
    )
    switch_mutation_chance = mutation_chance if mutation_type == "switch" else 0.0

    all_results = []
    for _ in range(n_runs):
        result = run_genetic_algorithm(
            data,
            n_generations=n_generations,
            cross_aggregating=cross_aggregating,
            population_size=population_size,
            severity_of_mutation=severity_of_mutation,
            mutation_aggregation_chance=mutation_aggregation_chance,
            normal_mutation_chance=normal_mutation_chance,
            switch_mutation_chance=switch_mutation_chance,
            tournament_size=tournament_size,
            survivors=survivors,
        )
        all_results.append(result)

    all_results = np.array(all_results)
    median_results = np.median(all_results, axis=0)
    std_results = np.std(all_results, axis=0)

    return median_results, std_results


def plot_results(title, values, medians, stds):
    plt.figure(figsize=(10, 5))

    colors = ["r", "g", "b", "y", "m", "c"]
    for i, value in enumerate(values):
        plt.plot(medians[i], label=value, color=colors[i])
        plt.fill_between(
            range(100),
            medians[i] - stds[i],
            medians[i] + stds[i],
            alpha=0.2,
            color=colors[i],
        )

    plt.xlabel("Generation")
    plt.ylabel("Cost")
    plt.yscale("log")
    plt.title("Genetic Algorithm Results")
    plt.legend(title=title)
    plt.show()


def run_differential_algorithm(
    data,
    n_generations: int = 250,
    diff_F: float = 1,
    diff_CR: float = 0.8,
    num_of_chunks: int = 40
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
        diff_F=diff_F,
        diff_CR=diff_CR,
    )

    result = []
    algorithm.generate_genes()

    for i in range(n_generations):
        print(f"generation {i} ")
        generation = algorithm.run_generation()
        costs = [algorithm.evaluate_cost(gene) for gene in generation]
        generation_min = min(costs)
        result.append(generation_min)

        # TODO: Remove this
        # generation_max = max(costs)
        # generation_avg = sum(costs)/len(costs)
        # print(f"Generation {i}:")
        # print(f"\tMin: {generation_min}, Max: {generation_max}, Avg: {generation_avg}")

    return result


if __name__ == "__main__":
    with open("data.txt", "r") as file:
        file_content = file.read()
    data = parse_sndlib_file(file_content)

    result = run_differential_algorithm(data)
    print(f"Result: {result}")
