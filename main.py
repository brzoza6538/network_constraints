from data_reader import parse_sndlib_file
from algorithms.genetic import EvolutionAlgorithm

with open("data.txt", "r") as file:
    file_content = file.read()

nodes, links, demands, admissible_paths = parse_sndlib_file(file_content)

x = EvolutionAlgorithm(nodes, links, demands, admissible_paths)
x.generate_genes()










for i in range (100):
    generation = x.run_generation()
    min = 99999999
    for gene in generation:
        score = x.evaluate_cost(gene)
        if(score < min):
            min = score
    print(min)





