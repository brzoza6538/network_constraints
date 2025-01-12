import random 
import math


def rand_split(number, num_of_parts):
    '''
        Randomly splits a number into num_of_parts segments.
        Segments can have size zero. num_of_parts must be <= the number of possible split points.

        w jaki sposób???
    '''
    possible_splits = list(range(0, number + 1, 5))
    
    splits = sorted(random.sample(possible_splits, num_of_parts - 1))
    splits = [0] + splits + [number]
    
    parts = [splits[i+1] - splits[i] for i in range(len(splits) - 1)]
    
    # parts = [0] * (num_of_parts)
    # parts[random.randint(0, num_of_parts-1)] += int((number/20))
    # parts[random.randint(0, num_of_parts-1)] += int((number/20))


    return parts




class GeneticAlgorithm:

    def __init__(self, nodes, links, demands, admissible_paths, aggregation=False):
        self._nodes = nodes
        self._links = links
        self._demands = demands
        self._admissible_paths = admissible_paths

        self._aggregation = aggregation

        self._population_size = 1000
        self._population = []
        self._punishment_for_overuse = 1000


    def generate_gene(self):
        for i in range(self._population_size):
            genes = {}

            for demand in self._admissible_paths.keys():
                genes[demand] = {}
                splits = rand_split( self._demands[demand]["demand_value"], len(self._admissible_paths[demand]))
                for path in self._admissible_paths[demand].keys():
                    genes[demand][path] = splits.pop()
            self._population.append(genes)


    def evaluate_cost(self, gene):
        full_cost = 0
        link_usage = {}

        for link in self._links.keys():
            full_cost += self._links[link]["setup_cost"]
            link_usage[link] = 0

        for demand in self._admissible_paths.keys():
            for path in self._admissible_paths[demand].keys():
                for link in self._admissible_paths[demand][path]:
                    link_usage[link] += gene[demand][path]
                    # print(f"{link}    --   {path}   ----    {demand}   ------   {gene[demand][path]}")


        for link in link_usage.keys():

            # print(f"-----------{link_usage[link]}") 
            for capacity, cost in self._links[link]["modules"]:
                
                link_usage[link] -= capacity
                full_cost += cost

                # print(f"{link_usage[link]}   |\|   {capacity}   |\|   {cost}") 
                if(link_usage[link] <= 0):
                    break
            
            if(link_usage[link] > 0):
                full_cost += self._punishment_for_overuse

            # print(full_cost)        
        return full_cost


