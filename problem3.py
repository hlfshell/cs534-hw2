from math import floor
from time import time
from typing import List
from search import (
    Problem,
    UndirectedGraph,
    Node,
    argmax_random_tie,
    fitness_threshold,
    select,
)
import random
import numpy as np

class TSP_problem(Problem):

    """ subclass of Problem to define various functions """

    def two_opt(self, state):
        """ Neighbour generating function for Traveling Salesman Problem """
        neighbour_state = state[:]
        left = random.randint(0, len(neighbour_state) - 1)
        right = random.randint(0, len(neighbour_state) - 1)
        if left > right:
            left, right = right, left
        neighbour_state[left: right + 1] = reversed(neighbour_state[left: right + 1])
        return neighbour_state

    def actions(self, state):
        """ action that can be excuted in given state """
        return [self.two_opt]

    def result(self, state, action):
        """  result after applying the given action on the given state """
        return action(state)

    def path_cost(self, c, state1, action, state2):
        """ total distance for the Traveling Salesman to be covered if in state2  """
        cost = 0
        for i in range(len(state2) - 1):
            cost += distances[state2[i]][state2[i + 1]]
        cost += distances[state2[0]][state2[-1]]
        return cost

    def value(self, state):
        """ value of path cost given negative for the given state """
        return -1 * self.path_cost(None, None, None, state)

romania_map = UndirectedGraph(dict(
    Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
    Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
    Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
    Drobeta=dict(Mehadia=75),
    Eforie=dict(Hirsova=86),
    Fagaras=dict(Sibiu=99),
    Hirsova=dict(Urziceni=98),
    Iasi=dict(Vaslui=92, Neamt=87),
    Lugoj=dict(Timisoara=111, Mehadia=70),
    Oradea=dict(Zerind=71, Sibiu=151),
    Pitesti=dict(Rimnicu=97),
    Rimnicu=dict(Sibiu=80),
    Urziceni=dict(Vaslui=142)))

romania_map.locations = dict(
    Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
    Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
    Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
    Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
    Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
    Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
    Vaslui=(509, 444), Zerind=(108, 531))



distances = {}
all_cities = []

for city in romania_map.locations.keys():
    distances[city] = {}
    all_cities.append(city)

for name_1, coordinates_1 in romania_map.locations.items():
        for name_2, coordinates_2 in romania_map.locations.items():
            distances[name_1][name_2] = np.linalg.norm(
                [coordinates_1[0] - coordinates_2[0], coordinates_1[1] - coordinates_2[1]])
            distances[name_2][name_1] = np.linalg.norm(
                [coordinates_1[0] - coordinates_2[0], coordinates_1[1] - coordinates_2[1]])
    
all_cities.sort()
print("All cities:")
print(all_cities)
print("------------------")

print("3a")
print("------------------")

def hill_climbing(problem):
    
    """From the initial node, keep choosing the neighbor with highest value,
    stopping when no neighbor is better. [Figure 4.2]"""
    
    def find_neighbors(state, number_of_neighbors=100):
        """ finds neighbors using two_opt method """
        
        neighbors = []
        
        for i in range(number_of_neighbors):
            new_state = problem.two_opt(state)
            neighbors.append(Node(new_state))
            state = new_state
            
        return neighbors

    # as this is a stochastic algorithm, we will set a cap on the number of iterations
    iterations = 10000
    
    current = Node(problem.initial)
    while iterations:
        neighbors = find_neighbors(current.state)
        if not neighbors:
            break
        neighbor = argmax_random_tie(
            neighbors,
            key=lambda node: problem.value(node.state)
        )
        if problem.value(neighbor.state) >= problem.value(current.state):
            """Note that it is based on negative path cost method"""
            current.state = neighbor.state
        iterations -= 1

    return current.state


tsp = TSP_problem(all_cities)

start = time()
hill_solution = hill_climbing(tsp)
end = time()

hill_duration = end-start

print("Hill climbing solution:")
print(hill_solution)


print("------------------")
print ("3b")
print("------------------")

def genetic_algorithm(population, fitness_fn, ngen=1000, pmut=0.1, early_term=False):
    last_best = None
    last_best_count = 0
    carryover = 5
    population_size = len(population)
    early_termination_repetition = 10

    for i in range(ngen):
        population.sort(key=fitness_fn)
        new_population = population[0:carryover]

        while len(new_population) < population_size:
            fitness_scores = [fitness_fn(x) for x in population]
            a = random.choices(population, weights=fitness_scores)[0]
            b = random.choices(population, weights=fitness_scores)[0]
            child = mutate(crossover(a, b), pmut)
            new_population.append(child)

        
        fittest = population[0]

        # This is a form of early termination - if we find the optimal or
        # a great answer that isn't beaten in 3 generations, let's just
        # go with that and call it a day.
        if last_best == fittest and early_term:
            last_best_count += 1
            if last_best_count >= early_termination_repetition:
                print("Early termination hit")
                return fittest
        else:
            last_best = fittest
            last_best_count = 0
        
        population = new_population

    return population[0]


def crossover(x: List[str], y: List[str]):
    n = len(x)
    index = floor(n/2)

    new = x[0:index]
    used = x[0:index]

    for char in y:
        if char in used:
            continue
        new.append(char)
    
    return new


def mutate(x: List[str], pmut: float):
    """
    Checks if the mutation occurs. If it does, swap
    two random city positions
    """
    
    if random.uniform(0, 1) >= pmut:
        return x

    index_1 = random.randint(0, len(x)-1)
    index_2 = random.randint(0, len(x)-1)

    tmp = x[index_1]
    x[index_1] = x[index_2]
    x[index_2] = tmp

    return x


def init_population(pop_number, gene_pool):
    """Initializes population for genetic algorithm
    pop_number  :  Number of individuals in population
    gene_pool   :  List of possible values for individuals
    """
    g = len(gene_pool)
    population = []
    for i in range(pop_number):
        individual: List[str] = gene_pool.copy()
        random.shuffle(individual)
        population.append(individual)

    return population


def fitness_fn(sample: List[str]):
    """
    Our fitness function is the length of all the distances traveled.
    Note that the salesman has to again return to the origin city at
    the end of their path.
    """
    total_distance = 0
    traverse = sample.copy()
    current = traverse.pop(0)
    start = current
    while len(traverse) > 0:
        next = traverse.pop()
        total_distance += distances[current][next]
        current = next

    total_distance += distances[current][start]

    return total_distance
        

max_population = 100
mutation_rate = 0.07

population = init_population(max_population, all_cities)

start = time()
fittest = genetic_algorithm(population, fitness_fn, pmut=mutation_rate)
end = time()
ga_duration = end - start

print("Genetic Algorithm best result:")
print(fittest)
print(f"Fitness score: {fitness_fn(fittest)}")

print("Now with early termination")
start = time()
fittest_early_term = genetic_algorithm(population, fitness_fn, pmut=mutation_rate, early_term=True)
end = time()
ga_early_term_duration = end - start

print("Genetic Algorithm (w early term)best result:")
print(fittest_early_term)
print(f"Fitness score: {fitness_fn(fittest_early_term)}")


print("------------------")
print ("3c")
print("------------------")
print("Comparing Genentic Algorithm and Hill Climbing approaches:")

print(f"Genetic Alogirthm time taken: {ga_duration} seconds")
print(f"Genetic Alogirthm (early term) time taken: {ga_early_term_duration} seconds")
print(f"Hill Climbing time taken: {hill_duration} seconds")
print(f"Fitness of genetic algorithm best: {fitness_fn(fittest)}")
print(f"Fitness of genetic algorithm (early term) best: {fitness_fn(fittest_early_term)}")
print(f"Fitness of hill climbing best: {fitness_fn(hill_solution)}")
