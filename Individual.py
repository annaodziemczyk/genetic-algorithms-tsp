

"""
Basic TSP Example
file: Individual.py
"""

import random
import math
import GA

class Individual:
    def __init__(self, _size, _data, _initialSolutionType):
        """
        Parameters and general variables
        """
        self.fitness    = 0
        self.genes      = []
        self.genSize    = _size
        self.data       = _data
        self.initialSolutionType = _initialSolutionType

        if _initialSolutionType == GA.InitialSolutionType.HEURISTIC:
            self.nearestNeighbourGeneration()
        else:
            self.randomGeneration()

    def randomGeneration(self):
        self.genes = list(self.data.keys())

        for i in range(0, self.genSize):
            n1 = random.randint(0, self.genSize - 1)
            n2 = random.randint(0, self.genSize - 1)
            tmp = self.genes[n2]
            self.genes[n2] = self.genes[n1]
            self.genes[n1] = tmp

    def nearestNeighbourGeneration(self):

        cities = list(self.data.keys())
        current = random.randint(0, self.genSize - 1)

        i = 0
        visited = []
        self.genes.append(cities[current])
        visited.append(current)
        shortest_distance = None
        closest_city = None

        while len(self.genes) < self.genSize:
            if i not in visited:
                distance = self.euclideanDistance(cities[current], cities[i])
                if shortest_distance is None or distance < shortest_distance:
                    shortest_distance = distance
                    closest_city = i

            if i == len(cities) - 1:
                self.genes.append(cities[closest_city])
                current = closest_city
                visited.append(closest_city)
                i = 0
                shortest_distance = None
                closest_city = None
            else:
                i += 1


    def setGene(self, genes):
        """
        Updating current choromosome
        """
        self.genes = []
        for gene_i in genes:
            self.genes.append(gene_i)

    def copy(self):
        """
        Creating a new individual
        """
        ind = Individual(self.genSize, self.data, self.initialSolutionType)
        for i in range(0, self.genSize):
            ind.genes[i] = self.genes[i]
        ind.fitness = self.getFitness()
        return ind

    def euclideanDistance(self, c1, c2):
        """
        Distance between two cities
        """
        d1 = self.data[c1]
        d2 = self.data[c2]
        return math.sqrt( (d1[0]-d2[0])**2 + (d1[1]-d2[1])**2 )

    def getFitness(self):
        return self.fitness

    def computeFitness(self):
        """
        Computing the cost or fitness of the individual
        """
        self.fitness    = self.euclideanDistance(self.genes[0], self.genes[len(self.genes)-1])
        for i in range(0, self.genSize-1):
            self.fitness += self.euclideanDistance(self.genes[i], self.genes[i+1])

