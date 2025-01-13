from data_reader import parse_sndlib_file
from algorithms.genetic import GeneticAlgorithm

with open("data.txt", "r") as file:
    file_content = file.read()

nodes, links, demands, admissible_paths = parse_sndlib_file(file_content)

x = GeneticAlgorithm(nodes, links, demands, admissible_paths)
x.generate_genes()

print(x.run_generation())






# gene_1 = x._population[0]


# for demand in gene_1:
#     if demand == "Demand_5_7" or demand == "Demand_3_4":
#         print(f"DEMAND: {demand}")
#         for path in gene_1[demand]:
#             print(f"\tPATH: {path}")
#             print(f"\t\tLINKS: {gene_1[demand][path]}")
#             #for link in admissible_paths[demand][path]:
#                 #print(f"\t\t\tLINNKS: {link}")


# gene_1 = x.mutate_without_aggregation(gene_1)


# for demand in gene_1:
#     if demand == "Demand_5_7" or demand == "Demand_3_4":
#         print(f"DEMAND: {demand}")
#         for path in gene_1[demand]:
#             print(f"\tPATH: {path}")
#             print(f"\t\tLINKS: {gene_1[demand][path]}")
#             #for link in admissible_paths[demand][path]:
#                 #print(f"\t\t\tLINNKS: {link}")










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
