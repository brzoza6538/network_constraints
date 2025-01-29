import random
import copy

# typy - aggregation, without aggregation


def rand_split(number, num_of_parts, number_of_splits):

    parts = [0] * (num_of_parts)
    half_a = int(number / number_of_splits)
    
    for _ in range(number_of_splits):
        parts[random.randint(0, num_of_parts - 1)] += half_a
    
    parts[random.randint(0, num_of_parts - 1)] += number - (number_of_splits * half_a)
    return parts



class GeneticAlgorithm:

    def __init__(
        self,
        nodes,
        links,
        demands,
        admissible_paths,
        cross_aggregating=True,
        num_of_splits=40, # czy zaczynamy z podziałem na wiele czy z podziałem na kilka grupek z możliwością nakładania się 

        population_size=1000,
        severity_of_mutation=0.5,

        mutation_aggregation_chance=0.5,
        normal_mutation_chance=0.5,
        switch_mutation_chance=0.5,

        tournament_size=2,
        survivors=20,
    ):
        self.print_uses = 0

        self._nodes = nodes
        self._links = links
        self._demands = demands
        self._admissible_paths = admissible_paths

        self._cross_aggregating = cross_aggregating
        self._best_to_survive = survivors
        self._population_size = population_size
        self._num_of_splits = num_of_splits 
        self._population = []
        self._punishment_for_overuse = 1000000

        self._severity_of_mutation = severity_of_mutation

        self._mutation_aggregation_chance = mutation_aggregation_chance
        self._normal_mutation_chance = normal_mutation_chance
        self._switch_mutation_chance = switch_mutation_chance

        self._mutation_aggregation_fadeout = 0.99
        self._normal_mutation_fadeout = 0.99
        self._switch_mutation_fadeout = 0.99

        self._tournament_size = tournament_size

    # TODO remove later
    def print_family(self, parent_1, parent_2, child_gene):
        for demand in self._admissible_paths.keys():
            print(demand)
            for path in self._admissible_paths[demand].keys():
                print(
                    f"\t{self._admissible_paths[demand][path]} \t-\t {parent_1[demand][path]} {parent_2[demand][path]} {child_gene[demand][path]}"
                )

    def generate_genes(self):
        for i in range(self._population_size):
            genes = {}

            for demand in self._admissible_paths.keys():
                genes[demand] = {}
                splits = rand_split(
                    self._demands[demand]["demand_value"],
                    len(self._admissible_paths[demand]),
                    self._num_of_splits
                )
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

        for link in link_usage.keys():
            for capacity, cost in self._links[link]["modules"]:
                link_usage[link] -= capacity
                full_cost += cost
                if link_usage[link] <= 0:
                    break
            if link_usage[link] > 0:
                full_cost += self._punishment_for_overuse

        return full_cost


    def cross_for_aggregation(self, gene_1, gene_2):
        """
        returns child - uses allels that send less thorugh less links (used links * send size)
        """
        child = copy.deepcopy(gene_1)

        for demand in self._admissible_paths.keys():
            score = 0
            for path in self._admissible_paths[demand].keys():
                score += gene_1[demand][path] * len(self._admissible_paths[demand][path])
                score -= gene_2[demand][path] * len(self._admissible_paths[demand][path])
            if score > 0:
                child[demand] = gene_2[demand]
        return child

    def cross_without_aggregation(self, gene_1, gene_2):
        """
        returns child - gets average spread of demands by allels
        """
        child = copy.deepcopy(gene_1)

        for demand in self._admissible_paths.keys():
            remaining_demand = self._demands[demand]["demand_value"]  # unikamy tracenia końcówki na na zaokrągleniu
            for i, path in enumerate(self._admissible_paths[demand]):
                child[demand][path] = int((gene_1[demand][path] + gene_2[demand][path]) / 2)
                remaining_demand -= child[demand][path]

                # Jeśli to ostatnia ścieżka, dodajemy resztę zapotrzebowania by uniknąć tracenia na ułamkach
                if i == len(self._admissible_paths[demand]) - 1:
                    child[demand][path] += remaining_demand

        return child

    def mutate_for_aggregation(self, gene):
        """
        Returns child - gets average spread of demands by allels
        """
        for demand in self._admissible_paths.keys(): # w innych przypadkach nie ma sensu, ale chyba tutaj każdy chromosom lepiej żeby przez o przeszedł
            demand = random.choice(list(self._admissible_paths.keys()))

            paths_to_steal_from = [
                path for path in gene[demand].keys() if gene[demand][path] > 0
            ]

            if paths_to_steal_from:
                path_to_steal_from = min(
                    paths_to_steal_from, key=lambda path: gene[demand][path]
                )

                paths_to_give_to = [
                    path for path in gene[demand].keys() if path != path_to_steal_from
                ]
                random_path_to_give = random.choice(paths_to_give_to)

                gene[demand][random_path_to_give] += gene[demand][path_to_steal_from]
                gene[demand][path_to_steal_from] = 0
        return gene

    def mutate_without_aggregation(self, gene):
        """
        Returns child - gets average spread of demands by allels
        """
        for demand in self._admissible_paths.keys(): # -  lepiej jedno czy wszystkie?

        # demand = random.choice(list(self._admissible_paths.keys()))
            amount_to_steal = int(self._demands[demand]["demand_value"] * self._severity_of_mutation)

            paths_to_steal_from = [path for path in gene[demand].keys() if gene[demand][path] > 0]

            if paths_to_steal_from:
                random_path_to_steal = random.choice(paths_to_steal_from)
                gene[demand][random_path_to_steal] -= amount_to_steal

                paths_to_give_to = [
                    path for path in gene[demand].keys() if path != random_path_to_steal
                ]

                random_path_to_give = random.choice(paths_to_give_to)
                gene[demand][random_path_to_give] += amount_to_steal

        return gene

    def mutate_with_switch(self, gene):
        """
        Returns child - gets average spread of demands by allels
        """
        for demand in self._admissible_paths.keys():
        # demand = random.choice(list(self._admissible_paths.keys()))

            amount_to_steal = int(self._demands[demand]["demand_value"])

            paths_to_steal_from = [
                path
                for path in gene[demand].keys()
                if gene[demand][path] >= amount_to_steal
            ]

            if paths_to_steal_from:
                random_path_to_steal = random.choice(paths_to_steal_from)
                help = gene[demand][random_path_to_steal]

                paths_to_give_to = [
                    path for path in gene[demand].keys() if path != random_path_to_steal
                ]

                random_path_to_give = random.choice(paths_to_give_to)
                gene[demand][random_path_to_steal] = gene[demand][random_path_to_give]
                gene[demand][random_path_to_give] += help

        return gene

    def tournament_selection(self):
        tournament = random.sample(self._population, self._tournament_size)
        best_gene = min(tournament, key=self.evaluate_cost)
        return best_gene

    def update_probabilities(self):
        self._mutation_aggregation_chance = self._mutation_aggregation_chance * self._mutation_aggregation_fadeout
        self._normal_mutation_chance = self._normal_mutation_chance * self._normal_mutation_fadeout
        self._switch_mutation_chance = self._switch_mutation_chance * self._switch_mutation_fadeout
                                        
                            
    def run_generation(self):
        new_population = sorted(self._population, key=self.evaluate_cost)[:self._best_to_survive]

        while len(new_population) < self._population_size:
            parent_1 = self.tournament_selection()
            parent_2 = self.tournament_selection()

            if(self._cross_aggregating == True):
                child_gene = self.cross_for_aggregation(parent_1, parent_2)
            else:
                child_gene = self.cross_without_aggregation(parent_1, parent_2)


            if random.random() < self._mutation_aggregation_chance:
                child_gene = self.mutate_for_aggregation(child_gene)
            if random.random() < self._switch_mutation_chance:
                child_gene = self.mutate_with_switch(child_gene)
            if random.random() < self._normal_mutation_chance:
                child_gene = self.mutate_without_aggregation(child_gene)

            new_population.append(child_gene)
        self._population = new_population
        self.update_probabilities()
        return self._population
