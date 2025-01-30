import numpy as np

from algorithms.genetic import GeneticAlgorithm
from data_reader import parse_sndlib_file
from algorithms.differential import DifferentialEvolutionAlgorithm

def run_genetic_algorithm(
    data,
    n_generations: int = 100,
    cross_aggregating: bool = True,
    severity_of_mutation: float = 0.8,
    mutation_aggregation_chance: float = 0.0,
    normal_mutation_chance: float = 0.8,
    switch_mutation_chance: float = 0.0,
    tournament_size: int = 2,
    differential: bool = False,
    num_of_splits: int = 40,
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
        num_of_splits=num_of_splits,
        cross_aggregating=cross_aggregating,
        severity_of_mutation=severity_of_mutation,
        mutation_aggregation_chance=mutation_aggregation_chance,
        normal_mutation_chance=normal_mutation_chance,
        switch_mutation_chance=switch_mutation_chance,
        tournament_size=tournament_size,
    )
    algorithm.generate_genes()

    result = []
    for i in range(n_generations):
        print(f"generation {i} ")

        generation = (
            algorithm.differential_run_generation()
            if differential
            else algorithm.run_generation()
        )

        costs = ([algorithm.evaluate_cost(gene) for gene in generation])
        generation_min = min(costs)
        generation_max = max(costs)
        generation_avg = sum(costs)/len(costs)
        print(f"{generation_min} - {generation_max} - {generation_avg}")

        result.append(generation_min)

    return result


def test_genetic_algorithm(
    data,
    n_runs: int = 10,
    n_generations: int = 100,
    cross_aggregating: bool = False,
    severity_of_mutation: float = 0.5,
    mutation_aggregation_chance: float = 0.1,
    normal_mutation_chance: float = 0.2,
    switch_mutation_chance: float = 0.3,
    tournament_size: int = 2,
    differential: bool = False,
    num_of_splits: int = 40,

) -> tuple:
    all_results = []
    for _ in range(n_runs):
        result = run_genetic_algorithm(
            data,
            num_of_splits=num_of_splits,
            n_generations=n_generations,
            cross_aggregating=cross_aggregating,
            severity_of_mutation=severity_of_mutation,
            mutation_aggregation_chance=mutation_aggregation_chance,
            normal_mutation_chance=normal_mutation_chance,
            switch_mutation_chance=switch_mutation_chance,
            tournament_size=tournament_size,
            differential=differential,
        )
        all_results.append(result)

    all_results = np.array(all_results)
    mean_results = np.mean(all_results, axis=0)
    std_results = np.std(all_results, axis=0)

    return mean_results, std_results



def run_differential_algorithm(
    data,
    n_generations: int = 250,
    diff_F: float = 1,
    diff_CR: float = 0.8,
    num_of_splits: int = 40
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
        diff_CR = diff_CR,
        num_of_splits=num_of_splits,
    )

    result = []
    algorithm.generate_genes()

    for i in range(n_generations):
        print(f"generation {i} ")
        generation = (algorithm.run_generation())
        costs = ([algorithm.evaluate_cost(gene) for gene in generation])
        generation_min = min(costs)
        generation_max = max(costs)
        generation_avg = sum(costs)/len(costs)

        print(f"{generation_min} - {generation_max} - {generation_avg}")

        result.append(generation_min)

    return result



if __name__ == "__main__":
    with open("data.txt", "r") as file:
        file_content = file.read()
    data = parse_sndlib_file(file_content)

    # result = run_differential_algorithm(data)
    # print(f"Result: {result}")

    result = run_genetic_algorithm(data)
    print(f"Result: {result}")
