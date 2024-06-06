import sys
import numpy as np
import random
import operator

## City
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city) :
        x_dis = abs(self.x - city.x)
        y_dis = abs(self.y - city.y)
        distance = np.sqrt((x_dis **2 + y_dis **2))
        return distance
    
    def __repr__(self):
        return "({0},{1})".format(self.x, self.y)

## Fitness
class Fitness:
    def __init__(self, route):
       self.route = route
       self.distance = 0
       self.fitness = 0 
    
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

    ## TODO here


    return selection_res


def main() -> int:
    return 0

if __name__ == "__main__":
    sys.exit(main())