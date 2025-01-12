from data_reader import parse_sndlib_file
from algorithms.genetic import GeneticAlgorithm

with open("data.txt", "r") as file:
    file_content = file.read()

nodes, links, demands, admissible_paths = parse_sndlib_file(file_content)

x = GeneticAlgorithm(nodes, links, demands, admissible_paths)
x.generate_gene()




last = 0
for gene in x._population:
    eva = (x.evaluate_cost(gene))
    if abs(eva - last) > 1000:
        print(f'{eva} ---- {last}') 
        last = eva







# for demand in x._population[0]:
#     if demand == "Demand_5_7":
#         print(f"DEMAND: {demand}")
#         for path in x._population[0][demand]:
#             print(f"\tPATH: {path}")
#             print(f"\t\tLINKS: {x._population[0][demand][path]}")
#             for link in admissible_paths[demand][path]:
#                 print(f"\t\t\tLINNKS: {link}")





# for demand in admissible_paths.keys():
#         print(demand)
#         print (y[demand])
#         print (demands[demand])
#         print("\n-------------\n")






# print(links[next(iter(links.keys()))]["setup_cost"])


# modules = links[next(iter(links.keys()))]["modules"]
# print(modules)

# for i in range(0, len(modules), 2):
#     print(f"{modules[i]} -- {modules[i+1]}")





# Wyświetlenie wyniku
# print(nodes[next(iter(nodes.keys()))])
# print("\n\n\n\n----------------------------\n\n\n")
# print(links[next(iter(links.keys()))])
# print("\n\n\n\n----------------------------\n\n\n")
# print(demands[next(iter(demands.keys()))])
# print("\n\n\n\n----------------------------\n\n\n")
# print(admissible_paths[next(iter(admissible_paths.keys()))])
# print("\n\n\n\n----------------------------\n\n\n")
