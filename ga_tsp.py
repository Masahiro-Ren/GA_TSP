import sys
import numpy as np
import random
import operator
import pandas as pd
import wandb

from python_tsp.exact import solve_tsp_dynamic_programming

## City
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city) :
        x_dis = abs(self.x - city.x)
        y_dis = abs(self.y - city.y)
        distance = np.sqrt((x_dis **2) + (y_dis **2))
        return distance
    
    def __repr__(self):
        return "({0},{1})".format(self.x, self.y)

## Fitness
class Fitness:
    def __init__(self, route):
       self.route = route
       self.distance = 0
       self.fitness = 0.0 
    
    def routeDistance(self):
        if self.distance == 0:
            path_dis = 0
            for i in range(0, len(self.route)):
                from_city = self.route[i]
                to_city = None

                if i + 1 < len(self.route):
                    to_city = self.route[i + 1]
                else:
                    to_city = self.route[0]
                path_dis += from_city.distance(to_city)
            self.distance = path_dis
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.routeDistance()
            # if self.distance != 0:
            #     self.fitness = 1.0 / float(self.distance)
            self.fitness = 1.0 / float(self.distance)
        return self.fitness

## Create random route
def createRoute(city_list):
    route = random.sample(city_list, len(city_list))
    return route

## Initialize population (a set of random route)
def initialPopulation(pop_size, city_list):
    population = []

    for i in range(0, pop_size):
        population.append(createRoute(city_list))
    
    return population

def rankRoutes(population):
    fitness_res = {}
    for i in range(0, len(population)):
        fitness_res[i] = Fitness(population[i]).routeFitness()
    
    return sorted(fitness_res.items(), key=operator.itemgetter(1), reverse=True)

def selection(ranked_pop, num_elites):
    selection_res = []

    df = pd.DataFrame(np.array(ranked_pop), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, num_elites):
        selection_res.append(ranked_pop[i][0])
    
    for i in range(0, len(ranked_pop) - num_elites):
        rand = 100 * random.random()
        for j in range(0, len(ranked_pop)):
            if rand <= df.iat[j,3]:
                selection_res.append(ranked_pop[j][0])
                break 

    return selection_res

def matching(population, selection_res):
    matching_pool = []
    for i in range(0, len(selection_res)):
        index = selection_res[i]
        matching_pool.append(population[index])
    
    return matching_pool

def crossover(parent1, parent2):
    
    child = []
    chromo1 = []
    chromo2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    gene_start = min(geneA, geneB)
    gene_end = max(geneA, geneB)

    for i in range(gene_start, gene_end):
        chromo1.append(parent1[i])
        
    chromo2 = [item for item in parent2 if item not in chromo1]

    child = chromo1 + chromo2

    return child

def crossoverPop(matching_pool, num_elites):
    children = []
    length = len(matching_pool) - num_elites
    pool = random.sample(matching_pool, len(matching_pool))

    for i in range(0, num_elites):
        children.append(matching_pool[i])
    
    for i in range(0, length):
        p1 = i
        p2 = len(matching_pool) - i - 1
        child = crossover(pool[p1], pool[p2])
        children.append(child)
    return children

def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if(random.random() < mutation_rate):
            j = int(random.random() * len(individual))

            gen1 = individual[i]
            gen2 = individual[j]
            individual[i] = gen2
            individual[j] = gen1

    return individual

def mutatePop(population, mutation_rate):
    mutated_pop = []

    for i in range(0, len(population)):
        ind = mutation(population[i], mutation_rate)
        mutated_pop.append(ind)
    return mutated_pop

def getNextGeneration(cur_gen, num_elites, mutation_rate):
    ranked_pop = rankRoutes(cur_gen)
    selection_res = selection(ranked_pop, num_elites)
    matching_pool = matching(cur_gen, selection_res)
    children = crossoverPop(matching_pool, num_elites)
    next_gen = mutatePop(children, mutation_rate)

    return next_gen
            
def GA_TSP(city_list, num_pop, num_elites, mutation_rate, epochs):
    pop = initialPopulation(num_pop, city_list)

    print("Initial distance : " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, epochs):
        pop = getNextGeneration(pop, num_elites, mutation_rate)
        # cur_dist = 1 / rankRoutes(pop)[0][1]
        # wandb.log({"distance":cur_dist})
    
    print("Final distance : " + str(1 / rankRoutes(pop)[0][1]))
    best_route_idx = rankRoutes(pop)[0][0]
    return pop[best_route_idx]

# Calculate refernce answer
def refernce_ans(city_list):
    size = len(city_list)

    distance_mat = np.zeros((size, size))

    for i in range(0, size):
        for j in range(0, size):
            if i == j :
                continue
            distance_mat[i][j] = city_list[i].distance(city_list[j])
    
    permutation, distance = solve_tsp_dynamic_programming(distance_mat)
    print("Verified distance: {0}".format(distance))

    for idx in permutation:
        print(city_list[idx])

def main() -> int:
    
    num_city = 25
    num_pop = 100
    num_elites = 20
    mutation_rate = 0.01
    epochs = 500

    city_list = []
    for i in range(0, num_city):
        rand_x = int(random.random() * 200)
        rand_y = int(random.random() * 200)
        city_list.append(City(rand_x, rand_y))

    # wandb.init()

    # in big problem size, reference_ans can take very very ...... long time
    # reference_ans(city_list)

    best_route = GA_TSP(city_list, num_pop, num_elites, mutation_rate, epochs)

    for i in range(0, len(best_route)):
        print(best_route[i])

    # wandb.finish()
    return 0

if __name__ == "__main__":
    sys.exit(main())