

"""
Author: Anna Odziemczyk R00132719
file:
Rename this file to TSP_x.py where x is your student number 
"""

from GA import *
from Individual import *
import sys
import logging
from Configuration import *
from Profiler import profile
import heapq
from DataAnalytics import *
import copy
import numpy as np

# R00132719
myStudentNum = 132719 # Replace 12345 with your student number
random.seed(myStudentNum)

class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations, _configuration, _logLevel=logging.ERROR):

        """
        Setup logging
        logging.ERROR - log to console
        logging.DEBUG - log to debug file
        default logging level - logging.DEBUG
        """
        self.logger = logging.getLogger('GA')
        self.logger.setLevel(_logLevel)
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        fh = logging.FileHandler('debug.log')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        """
        Parameters and general variables
        """
        self.population     = []
        self.newpopulation  = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = _popSize
        self.genSize        = None
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0
        self.fName          = _fName
        self.data           = {}
        self.selectionType  = _configuration.selection
        self.initialSolutionType = _configuration.initial_solution
        self.crossoverType = _configuration.crossover
        self.mutationType = _configuration.mutation

        self.readInstance()
        self.initPopulation()

    def __str__(self):
        """
        outputs basic configuration for the run
        """
        return "\nFile name: " + self.fName \
               + "\nPopulation size: " + str(self.popSize) \
               + "\nMutation rate: " + str(self.mutationRate) \
               + "\nTotal iterations: " + str(self.maxIterations)

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
            individual = Individual(self.genSize, self.data, self.initialSolutionType)
            self.population.append(individual)

        """
        Setting the initial best solution to first individual in the population
        """
        self.best = self.population[0].copy()

        """
        converting the population to priority queue. First element has always the smallest fitness
        """
        heapq.heapify(self.population)

        print ("Best initial sol: ", self.best.getFitness())


    def updateBest(self, candidate):
        """
        adding the candidate to newpopulation and updaing the best solution
        :param candidate:
        :return:
        """
        heapq.heappush(self.newpopulation, candidate)

        if self.best.getFitness() == None or candidate.getFitness() < self.best.getFitness():
            # self.best = candidate.copy()
            self.best = candidate.copy()
            print ("iteration: ", self.iteration, "best: ", self.best.getFitness())

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.popSize-1) ]
        indB = self.matingPool[ random.randint(0, self.popSize-1) ]
        return [indA, indB]

    # @profile
    def stochasticUniversalSampling(self):
        """
        Your stochastic universal sampling Selection Implementation
        """
        #sort population by fitness
        sorted = heapq.nsmallest(self.popSize, range(self.popSize), self.population.__getitem__)
        #get the largest fitness score and add 1 to it
        largest = heapq.nlargest(1, self.population)[0].getFitness() + 1
        #convert the fitness scores in order to increase the probability of smaller fitness to be selected (smallest fitness becomes the largest)
        population = dict((i, largest - self.population[i].getFitness()) for i in sorted)

        #total fitness of the population
        F = sum(population.values())
        #number of individuals to be selected
        N = self.popSize
        #distance between pointers
        P = F/N

        #Select a random starting point on the wheel
        start = random.uniform(0, P)
        pointers = [start + i * P for i in range(0, N)]
        total = 0
        pointers.sort(reverse=True)

        fitnessMeasures = list(population.values())
        indeces = list(population.keys())

        #select individuals from the population
        keep = []
        for point in pointers:
            i = 0
            #continue updating total of individuals until the fitness point is not reached
            total += fitnessMeasures[0]
            while total < point:
                if i+1 == self.popSize:
                    total = F - total
                else:
                    i += 1
                    #continue going over the wheel when reached the end
                    total += fitnessMeasures[i]
            #select individual
            keep.append(indeces[i])

        # update mating pool with the individuals for selected indices
        self.matingPool = np.array(list(self.population))[keep]

    # @profile
    def uniformOrderBasedCrossover(self, indA, indB):
        #make sure the generation size for both indviduals and select the smaller one, if it's not the case
        size = min(len(indA.genes), len(indB.genes))
        #generate a binary template in the size of generation
        template = [random.randint(0, 1) for _ in range(size)]
        unpopulated = []

        #find positions of 0's in binary template
        for idx, val in enumerate(template):
            if val == 0:
                unpopulated.append(idx)

        #find genes in parent in positions of 0's in binary template
        sortForChildA = np.array(indA.genes)[unpopulated]
        sortForChildB = np.array(indB.genes)[unpopulated]

        #sort genes in order as they appear in parent
        sortForChildA = self.sortItems(sortForChildA, indB.genes)
        sortForChildB = self.sortItems(sortForChildB, indA.genes)

        #replace genes in selected for indicies corresponding to 0's in binary template in order as they occur in parent
        j = 0
        for index in unpopulated:
            indA.genes[index] = sortForChildA[j]
            indB.genes[index] = sortForChildB[j]
            j += 1

        return indA, indB

    def sortItems(self, childItems, parent):
        """
        sort child genes in order as they appear in parent
        :param childItems:
        :param parent:
        :return:
        """
        childAMap = {}

        #map index position of gene in parent to child gene
        for gene in childItems:
            childAMap[int(np.where(parent==gene)[0][0])] = gene

        #order genes by order as they occur in parent
        childAMap = dict(sorted(childAMap.items()))

        return list(childAMap.values())

    # @profile
    def pmxCrossover(self, indA, indB):
        """
        Your PMX Crossover Implementation
        """

        #make sure the generation size for both indviduals and select the smaller one, if it's not the case
        size = min(len(indA.genes), len(indB.genes))

        # pick 2 index numbers for gene swap
        gene_range = random.sample(range(0, size), 2)

        #sort genes
        gene_range.sort()

        childA = indA.copy()
        childB = indB.copy()

        # swap the sequence of genes within the index range
        childA.genes[gene_range[0]:gene_range[1]] = indB.genes[gene_range[0]:gene_range[1]]
        childB.genes[gene_range[0]:gene_range[1]] = indA.genes[gene_range[0]:gene_range[1]]

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
        """
        replace gene in child given the insert position is not in the range of genes already copied. Otherwise find new position
        by finding index of gene currently mapped to
        :param parent1:
        :param parent2:
        :param target: child
        :param source_gene:
        :param target_gene:
        :param start_swap_index:
        :param end_swap_index:
        :return:
        """
        mapping_index = parent1.genes.index(target_gene)
        if mapping_index < start_swap_index or mapping_index >= end_swap_index:
            target.genes[mapping_index] = source_gene
        else:
            self.swapGenesInSequence(parent1, parent2, target, source_gene, parent2.genes[parent1.genes.index(target_gene)], start_swap_index, end_swap_index)

        return target

    # @profile
    def reciprocalExchangeMutation(self, ind):
        """
        Your Reciprocal Exchange Mutation implementation
        """
        # pick 2 index numbers for gene swap
        gene1_index, gene2_index = self.pickRandomGeneIndeces()

        # swap genes
        ind.genes[gene1_index], ind.genes[gene2_index] = ind.genes[gene2_index], ind.genes[gene1_index]

        ind.computeFitness()
        self.updateBest(ind)

    # @profile
    def inversionMutation(self, ind):
        """
        Your Inversion Mutation implementation
        """
        # pick 2 index numbers for gene positions
        indexA, indexB = self.pickRandomGeneIndeces()
        generange = [indexA, indexB]

        generange.sort()

        # reverse order of genes in the index range
        ind.genes[generange[0]:generange[1]] = ind.genes[generange[0]:generange[1]][::-1]

        ind.computeFitness()
        self.updateBest(ind)

    def pickRandomGeneIndeces(self):
        if random.random() > self.mutationRate:
            self.pickRandomGeneIndeces()

        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        return indexA, indexB

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

    # def mutation(self, ind):
    #     """
    #     Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
    #     """
    #     indexA, indexB = self.pickRandomGeneIndeces()
    #
    #     tmp = ind.genes[indexA]
    #     ind.genes[indexA] = ind.genes[indexB]
    #     ind.genes[indexB] = tmp
    #
    #     ind.computeFitness()
    #     self.updateBest(ind)

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []

        if self.selectionType.STOCHASTIC_UNIVERSAL_SAMPLING:
            self.stochasticUniversalSampling()
        else:
            for ind_i in self.population:
                self.matingPool.append( ind_i.copy() )

    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        for i in range(0, self.popSize):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            indA, indB = self.randomSelection()
            childA, childB = None, None

            if self.crossoverType == GA.CrossoverType.PMX:
                childA, childB = self.pmxCrossover(indA, indB)
            elif self.crossoverType == GA.CrossoverType.UNIFORM_ORDER_BASED:
                childA, childB = self.uniformOrderBasedCrossover(indA, indB)

            if self.mutationType == GA.MutationType.INVERSION_MUTATION:
                self.inversionMutation(childA)
                self.inversionMutation(childB)
            elif self.mutationType == GA.MutationType.RECIPROCAL_EXCHANGE:
                self.reciprocalExchangeMutation(childA)
                self.reciprocalExchangeMutation(childB)

        # top 10 percent
        ten_percent_index = int(len(self.newpopulation) * 0.1)
        self.population = []
        for i in range(0, ten_percent_index):
            self.population.append(heapq.heappop(self.newpopulation))

        self.population = self.population + self.newpopulation[0:self.popSize-ten_percent_index]
        heapq.heapify(self.population)

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        self.newGeneration()

    # @profile
    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        self.iteration = 0
        while self.iteration < self.maxIterations:
            self.GAStep()
            self.iteration += 1

        # print ("Total iterations: ",self.iteration)

        print ("Best Solution: ", self.best.getFitness())


if len(sys.argv) < 5:
    print ("Error - Incorrect input")
    print ("Expecting python BasicTSP.py [instance], population size, mutation rate, configuration number (1 of 8) ")
    sys.exit(0)

# files = ["TSPdata\inst-4.tsp", "TSPdata\inst-6.tsp", "TSPdata\inst-16.tsp"]
problem_file = sys.argv[1]
# population_sizes = [100, 200, 300, 400]
population_size = int(sys.argv[2])
# mutation_rates = [0.1, 0.2, 0.3, 0.4]
mutation_rate = float(sys.argv[3])
config_no = int(sys.argv[4])-1
number_of_test_iterations = 5
no_of_iterations = 500

configurations = [
    Configuration("1", GA.SelectionType.RANDOM, GA.InitialSolutionType.RANDOM, GA.CrossoverType.UNIFORM_ORDER_BASED,
                       GA.MutationType.INVERSION_MUTATION),
    Configuration("2", GA.SelectionType.RANDOM, GA.InitialSolutionType.RANDOM, GA.CrossoverType.PMX, GA.MutationType.RECIPROCAL_EXCHANGE),
    Configuration("3", GA.SelectionType.STOCHASTIC_UNIVERSAL_SAMPLING, GA.InitialSolutionType.RANDOM,
                              GA.CrossoverType.UNIFORM_ORDER_BASED, GA.MutationType.RECIPROCAL_EXCHANGE),
    Configuration("4", GA.SelectionType.STOCHASTIC_UNIVERSAL_SAMPLING, GA.InitialSolutionType.RANDOM,
                              GA.CrossoverType.PMX, GA.MutationType.RECIPROCAL_EXCHANGE),
    Configuration("5", GA.SelectionType.STOCHASTIC_UNIVERSAL_SAMPLING, GA.InitialSolutionType.RANDOM,
                              GA.CrossoverType.PMX, GA.MutationType.INVERSION_MUTATION),
    Configuration("6", GA.SelectionType.STOCHASTIC_UNIVERSAL_SAMPLING, GA.InitialSolutionType.RANDOM,
                              GA.CrossoverType.UNIFORM_ORDER_BASED, GA.MutationType.INVERSION_MUTATION),
    Configuration("7", GA.SelectionType.STOCHASTIC_UNIVERSAL_SAMPLING, GA.InitialSolutionType.HEURISTIC,
                  GA.CrossoverType.PMX, GA.MutationType.RECIPROCAL_EXCHANGE),
    Configuration("8", GA.SelectionType.STOCHASTIC_UNIVERSAL_SAMPLING, GA.InitialSolutionType.HEURISTIC,
                          GA.CrossoverType.UNIFORM_ORDER_BASED, GA.MutationType.INVERSION_MUTATION)
]

config = configurations[config_no]

exp = Experiment(config, problem_file, no_of_iterations, mutation_rate, population_size)
for test in range(0, number_of_test_iterations):
    print("Run: " + str(test + 1))
    ga = BasicTSP(problem_file, population_size, mutation_rate, no_of_iterations, config)
    print(ga)
    print(config)
    ga.search()
    exp.saveResult(str(test + 1), ga.best)

# da = DataAnalytics()
# da.drawChart("populationSize", {"mutationRate":mutation_rates[0]}, "Population size", "Effect of Population Size")
# da.drawChart("mutationRate", {"populationSize":population_sizes[0]}, "Mutation Rate", "Effect of Mutation Rate")
# da.drawChartByMutationType("mutationRate", {"populationSize":population_sizes[0]}, "Mutation Rate", "Comparision of Mutation Strategies")
