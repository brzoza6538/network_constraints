from algorithms.genetic import EvolutionAlgorithm
from data_reader import parse_sndlib_file


def run_evolution_algorithm(
    nodes, links, demands, admissible_paths, n_generations: int = 100
) -> int:
    """
    Runs the evolution algorithm.
    :param file_path: Path to the file with the SNDlib data.
    :param n_generations: Number of generations to run the algorithm.
    """

    algorithm = EvolutionAlgorithm(nodes, links, demands, admissible_paths)
    algorithm.generate_genes()

    result = float("inf")
    for i in range(n_generations):
        generation = algorithm.run_generation()
        generation_min = min([algorithm.evaluate_cost(gene) for gene in generation])

        print(f"Generation: {i}, minimum: {generation_min}")
        result = min(result, generation_min)

    return result


if __name__ == "__main__":
    with open("data.txt", "r") as file:
        file_content = file.read()
    nodes, links, demands, admissible_paths = parse_sndlib_file(file_content)

    result = run_evolution_algorithm(nodes, links, demands, admissible_paths, n_generations=20)
    print(f"Result: {result}")