

"""
Author:
file:
Rename this file to TSP_x.py where x is your student number 
"""

import random
from Individual import *
import sys


myStudentNum = 132719 # Replace 12345 with your student number
random.seed(myStudentNum)

class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations):
        """
        Parameters and general variables
        """

        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = _popSize
        self.genSize        = None
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0
        self.fName          = _fName
        self.data           = {}

        self.readInstance()
        self.initPopulation()


    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y))
        file.close()

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data)
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print ("Best initial sol: ", self.best.getFitness())

    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            print ("iteration: ",self.iteration, "best: ",self.best.getFitness())

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.popSize-1) ]
        indB = self.matingPool[ random.randint(0, self.popSize-1) ]
        return [indA, indB]

    def stochasticUniversalSampling(self):
        """
        Your stochastic universal sampling Selection Implementation
        """
        total_fitness = sum(individual.getFitness() for individual in self.population)
        point_distance = total_fitness / 2
        start_point = random.uniform(0, point_distance)
        points = [start_point + i * point_distance for i in range(self.popSize)]


        parents = set()
        while len(parents) < self.popSize:
            random.shuffle(self.population)
            i = 0
            while i < len(points) and len(parents) < self.popSize:
                j = 0
                subset_sum = 0
                while j < len(self.population):
                    subset_sum += self.population[j].fitness
                    if subset_sum > points[i]:
                        parents.add(self.population[j])
                        break
                    j += 1
                i += 1

        self.matingPool = list(parents)


    def uniformCrossover(self, indA, indB):
        """
        Your Uniform Crossover Implementation
        """
        crossprob = 0.1
        size = min(len(indA.genes), len(indB.genes))
        for i in range(size):
            if random.random() < crossprob:
                indA.genes[i], indB.genes[i] = indB.genes[i], indA.genes[i]

        return indA, indB

    def pmxCrossover(self, indA, indB):
        """
        Your PMX Crossover Implementation
        """
        size = min(len(indA.genes), len(indB.genes))

        # pick 2 index numbers for gene swap
        gene_range = random.sample(range(0, size), 2)

        gene_range.sort()

        childA = indA.copy()
        childB = indB.copy()

        # swap the sequence of genes withing the index range
        childA.genes[gene_range[0]:gene_range[1]] = indB.genes[gene_range[0]:gene_range[1]]
        childB.genes[gene_range[0]:gene_range[1]] = indB.genes[gene_range[0]:gene_range[1]]

        # for each gene in sequence
        for i in range(gene_range[0], gene_range[1]):
            # check if genes copied from parent B are not in parent A
            if childB.genes[i] not in childA.genes[gene_range[0]:gene_range[1]]:
                childA = self.swapGenesInSequence(indA, indB, childA, childB.genes[i], childA.genes[i], gene_range[0],
                                                  gene_range[1])

            # check if genes copied from parent A are not in parent B
            if childA.genes[i] not in childB.genes[gene_range[0]:gene_range[1]]:
                childB = self.swapGenesInSequence(indB, indA, childB, childA.genes[i], childB.genes[i], gene_range[0], gene_range[1])

        return childA, childB

    def swapGenesInSequence(self, parent1, parent2, target, source_gene, target_gene, start_swap_index, end_swap_index):
        mapping_index = parent1.genes.index(target_gene)
        if mapping_index < start_swap_index or mapping_index >= end_swap_index:
            target[mapping_index] = source_gene
        else:
            self.swapGenesInSequence(parent1, parent2, target, source_gene, parent2.genes[parent1.genes.index(target_gene)], start_swap_index, end_swap_index)

        return target


    def reciprocalExchangeMutation(self, ind):
        """
        Your Reciprocal Exchange Mutation implementation
        """
        # pick 2 index numbers for gene swap
        gene1_index, gene2_index = random.sample(range(0, len(ind.genes)), 2)

        # swap genes
        ind.genes[gene1_index], ind.genes[gene2_index] = ind.genes[gene2_index], ind.genes[gene1_index]

        ind.computeFitness()
        self.updateBest(ind)


    def inversionMutation(self, ind):
        """
        Your Inversion Mutation implementation
        """
        # pick 2 index numbers for gene positions
        generange = random.sample(range(0, len(ind.genes)), 2)

        generange.sort()

        # reverse order of genes in the index range
        ind.genes[generange[0]:generange[1]] = ind.genes[generange[0]:generange[1]][::-1]

        ind.computeFitness()
        self.updateBest(ind)

    def crossover(self, indA, indB):
        """
        Executes a 1 order crossover and returns a new individual
        """
        child = []
        tmp = {}

        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        for i in range(0, self.genSize):
            if i >= min(indexA, indexB) and i <= max(indexA, indexB):
                tmp[indA.genes[i]] = False
            else:
                tmp[indA.genes[i]] = True
        aux = []
        for i in range(0, self.genSize):
            if not tmp[indB.genes[i]]:
                child.append(indB.genes[i])
            else:
                aux.append(indB.genes[i])
        child += aux

        child_individual = indB.copy()
        child_individual.setGene(child)

        return child_individual

    def mutation(self, ind):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []
        self.stochasticUniversalSampling()
        # for ind_i in self.population:
        #     self.matingPool.append( ind_i.copy() )

    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            # self.stochasticUniversalSampling()
            indA, indB = self.randomSelection()
            childA, childB = self.pmxCrossover(indA, indB)
            self.mutation(childA)
            self.mutation(childB)

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        self.newGeneration()

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        self.iteration = 0
        while self.iteration < self.maxIterations:
            self.GAStep()
            self.iteration += 1

        print ("Total iterations: ",self.iteration)
        print ("Best Solution: ", self.best.getFitness())

if len(sys.argv) < 2:
    print ("Error - Incorrect input")
    print ("Expecting python BasicTSP.py [instance] ")
    sys.exit(0)


if __name__ == "__main__":
    problem_file = sys.argv[1]

    # ga = BasicTSP(sys.argv[1], 300, 0.1, 500)
    ga = BasicTSP(sys.argv[1], 5, 0.1, 100)
    ga.search()
