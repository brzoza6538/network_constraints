import random
import copy


#DE/rand/1/bin

def rand_split(number, num_of_parts, number_of_splits):

    parts = [0] * (num_of_parts)
    half_a = int(number / number_of_splits)
    
    for _ in range(number_of_splits):
        parts[random.randint(0, num_of_parts - 1)] += half_a
    
    parts[random.randint(0, num_of_parts - 1)] += number - (number_of_splits * half_a)
    return parts


class DifferentialEvolutionAlgorithm:
    def __init__(
        self,
        nodes,
        links,
        demands,
        admissible_paths,
        aggregation=False,
        population_size=100,
        diff_F=1, 
        diff_CR=0.8,
        parental_tournament_size = 1, # for 1 = random - possibly best for some reason
        survivors_amount = 10
    ):
        self._nodes = nodes
        self._links = links
        self._demands = demands
        self._admissible_paths = admissible_paths

        self._aggregation = aggregation
        self._population_size = population_size
        self._population = []
        self._punishment_for_overuse = 1000000
        self._num_of_splits = 1

        self._diff_CR = diff_CR
        self._diff_F = diff_F  
        self._parental_tournament_size = parental_tournament_size
        self._survivors_amount = survivors_amount

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

    def diffuse_negative_path(self, child, demand):
        lowest_path = min(child[demand], key=child[demand].get)
        while child[demand][lowest_path] < 0:
            difuser_path = random.choice(list(child[demand].keys()))
            child[demand][difuser_path] += child[demand][lowest_path]
            child[demand][lowest_path] = 0
            lowest_path = min(child[demand], key=child[demand].get)
        return child

    def mutation_with_crossover(self, gene_1, gene_2, gene_3):
        #każdy demand musi mieć stałą sume path równą oczekiwanej w demand wartości 
        # moduł czy max(0, ...) nie zadziała 
        # wektor nie może być mniejszy od zera 
        negative_allel_flag = False
        child = copy.deepcopy(gene_1)

        for demand in self._admissible_paths.keys():
            balance = 0
            if random.random() < self._diff_CR:
                for path in self._admissible_paths[demand].keys():
                    diff = gene_2[demand][path] - gene_3[demand][path]
                    vector = gene_1[demand][path] + int(self._diff_F * diff)
                    balance -= int(self._diff_F * diff)
                    if vector < 0:
                        negative_allel_flag = True

                    child[demand][path] = vector

            # round(-1.1) +  round(-1.4) + round(3.5) - zmiany się mogą nie zerować jeśli będą zaokrąglane
            # balance pilnuje by dodawane wektory na pewno się zerowały, sprawiając że suma path zawsze równa się demand

            difuser_path = random.choice(list(child[demand].keys()))
            child[demand][difuser_path] += balance

            #jeśli jakiś chromosom ma ujemną wartość - będzie **bardzo** często sie zdarzało
            if negative_allel_flag:
                #rozwiązanie zachowujące plus minus agregacje
                child = self.diffuse_negative_path(child, demand)

        return child

    def differential_tournament_selection(self, child, parent):
        p_cost = self.evaluate_cost(parent)
        c_cost = self.evaluate_cost(child)
        if(p_cost < c_cost):
            return parent
        else:
            return child


    def parental_tournament_selection(self):
        parent_1_idx = self.tournament_selection()
        parent_2_idx = self.tournament_selection(exclude=[parent_1_idx])
        parent_3_idx = self.tournament_selection(exclude=[parent_1_idx, parent_2_idx])

        return self._population[parent_1_idx], self._population[parent_2_idx], self._population[parent_3_idx]

    def tournament_selection(self, exclude=[]):
        available_genes = [i for i in range(len(self._population)) if i not in exclude]
        possible_parents = random.sample(available_genes, self._parental_tournament_size)
        best_index = min(possible_parents, key=lambda idx: self.evaluate_cost(self._population[idx]))

        return best_index

    def check_if_uniqe(self, new_population, candidate):
        for gene in new_population:
            for demand in gene.keys():
                for path in gene[demand].keys():     
                    if gene[demand][path] != candidate[demand][path]:
                        return True
        return False
        
    def run_generation(self):
        new_population = sorted(self._population, key=self.evaluate_cost)[:self._survivors_amount]

        while len(new_population) < self._population_size:
            parent_1, parent_2, parent_3 = self.parental_tournament_selection()

            possible_child = self.mutation_with_crossover(parent_1, parent_2, parent_3)
            child = self.differential_tournament_selection(possible_child, parent_1)

            if self.check_if_uniqe(new_population, child):
                new_population.append(child)
 

        self._population = new_population
        return self._population