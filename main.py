import numpy as np

from algorithms.genetic import EvolutionAlgorithm
from data_reader import parse_sndlib_file


def run_evolution_algorithm(
    data,
    n_generations: int = 20,
    aggregation: bool = False,
    severity_of_mutation: float = 0.5,
    mutation_aggregation_chance: float = 0.1,
    normal_mutation_chance: float = 0.2,
    switch_mutation_chance: float = 0.3,
    tournament_size: int = 2,
    differential: bool = False,
) -> list[int]:
    """
    Runs the evolution algorithm.
    :param file_path: Path to the file with the SNDlib data.
    :param n_generations: Number of generations to run the algorithm.
    """

    nodes, links, demands, admissible_paths = data
    algorithm = EvolutionAlgorithm(
        nodes,
        links,
        demands,
        admissible_paths,
        aggregation=aggregation,
        severity_of_mutation=severity_of_mutation,
        mutation_aggregation_chance=mutation_aggregation_chance,
        normal_mutation_chance=normal_mutation_chance,
        switch_mutation_chance=switch_mutation_chance,
        tournament_size=tournament_size,
    )
    algorithm.generate_genes()

    result = []
    for i in range(n_generations):
        generation = (
            algorithm.differential_run_generation()
            if differential
            else algorithm.run_generation()
        )
        generation_min = min([algorithm.evaluate_cost(gene) for gene in generation])

        result.append(generation_min)

    return result


def test_evolution_algorithm(
    data,
    n_runs: int = 10,
    n_generations: int = 20,
    aggregation: bool = False,
    severity_of_mutation: float = 0.5,
    mutation_aggregation_chance: float = 0.1,
    normal_mutation_chance: float = 0.2,
    switch_mutation_chance: float = 0.3,
    tournament_size: int = 2,
    differential: bool = False,
) -> tuple:
    all_results = []
    for _ in range(n_runs):
        result = run_evolution_algorithm(
            data,
            n_generations=n_generations,
            aggregation=aggregation,
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


if __name__ == "__main__":
    with open("data.txt", "r") as file:
        file_content = file.read()
    nodes, links, demands, admissible_paths = parse_sndlib_file(file_content)

    result = run_evolution_algorithm(
        nodes, links, demands, admissible_paths, n_generations=20, differential=True
    )
    print(f"Result: {result}")
