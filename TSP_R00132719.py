

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

# R00132719
myStudentNum = 132719 # Replace 12345 with your student number
random.seed(myStudentNum)

class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations, _configuration, _logLevel=logging.ERROR):
        """
        Parameters and general variables
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
        self.selectionType  = _configuration.selection
        self.initialSolutionType = _configuration.initial_solution
        self.crossoverType = _configuration.crossover
        self.mutationType = _configuration.mutation

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
            individual = Individual(self.genSize, self.data, self.initialSolutionType)
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print ("Best initial sol: ", self.best.getFitness())


    def updateBest(self, candidate):
        if self.best.getFitness() == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            print ("iteration: ",self.iteration, "best: ",self.best.getFitness())

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
        total_fitness = sum(individual.getFitness() for individual in self.population)
        point_distance = total_fitness / 2
        start_point = random.uniform(0, point_distance)
        points = [start_point + i * point_distance for i in range(self.popSize)]
        pointsSize = len(points)

        parents = set()
        parents_count = 0

        while parents_count < self.popSize:
            random.shuffle(self.population)
            i = 0
            while i < pointsSize and parents_count < self.popSize:
                j = 0
                subset_sum = 0
                while j < self.popSize:
                    subset_sum += self.population[j].fitness
                    if subset_sum > points[i]:
                        parents.add(self.population[j])
                        parents_count += 1
                        break
                    j += 1
                i += 1
            parents_count = len(parents)

        self.matingPool = list(parents)

    # def uniformCrossover(self, indA, indB):
    #     """
    #     Your Uniform Crossover Implementation
    #     """
    #     swapprob = 0.5
    #     size = min(len(indA.genes), len(indB.genes))
    #     for i in range(size):
    #         if random.random() < swapprob:
    #             indA.genes[i], indB.genes[i] = indB.genes[i], indA.genes[i]
    #
    #     return indA, indB

    # @profile
    def uniformOrderBasedCrossover(self, indA, indB):
        size = min(len(indA.genes), len(indB.genes))
        template = [random.randint(0, 1) for _ in range(size)]
        sortForChildA = []
        sortForChildB = []
        unpopulated = []

        i = 0
        for binary_value in template:
            if binary_value == 0:
                unpopulated.append(i)
                sortForChildA.append(indA.genes[i])
                sortForChildB.append(indB.genes[i])
            i += 1

        sortForChildA = self.sortItems(sortForChildA, indB.genes)
        sortForChildB = self.sortItems(sortForChildB, indA.genes)

        j = 0
        for index in unpopulated:
            indA.genes[index] = sortForChildA[j]
            indB.genes[index] = sortForChildB[j]
            j += 1

        return indA, indB

    def sortItems(self, childItems, parent):
        childAMap = {}

        for gene in childItems:
            childAMap[parent.index(gene)] = gene

        childAMap = dict(sorted(childAMap.items()))

        return list(childAMap.values())

    # @profile
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

    # @profile
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

    # @profile
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

        self.updateStartTime()
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

        self.printExecutionTime("crossover")

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
        newpopulation = []

        for i in range(0, self.popSize):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            # self.stochasticUniversalSampling()
            indA, indB = self.randomSelection()
            childA, childB = None, None

            if self.crossoverType == GA.CrossoverType.PMX:
                childA, childB = self.pmxCrossover(indA, indB)
            elif self.crossoverType == GA.CrossoverType.UNIFORM_ORDER_BASED:
                childA, childB = self.uniformOrderBasedCrossover(indA, indB)

            if self.mutationType == GA.MutationType.INVERSION_MUTATION:
                self.inversionMutation(childA)
                newpopulation.append(self.best)
                self.inversionMutation(childB)
                newpopulation.append(self.best)
            elif self.mutationType == GA.MutationType.RECIPROCAL_EXCHANGE:
                self.reciprocalExchangeMutation(childA)
                newpopulation.append(self.best)
                self.reciprocalExchangeMutation(childB)
                newpopulation.append(self.best)

        # self.population = newpopulation

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        self.newGeneration()

    @profile
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


if len(sys.argv) < 2:
    print ("Error - Incorrect input")
    print ("Expecting python BasicTSP.py [instance] ")
    sys.exit(0)

# problem_file = sys.argv[1]
# ga = BasicTSP(sys.argv[1], 300, 0.1, 500)

# files = ["TSPdata\inst-4.tsp", "TSPdata\inst-6.tsp", "TSPdata\inst-16.tsp"]
# population_sizes = [100, 200, 300, 400]
# mutation_rates = [0.1, 0.2, 0.3, 0.4]
# number_of_test_iterations = 5
# no_of_iterations = 500

files = ["TSPdata\inst-4.tsp"]
population_sizes = [100]
mutation_rates = [0.1]
number_of_test_iterations = 5
no_of_iterations = 500

# configurations = dict({
#     "1": Configuration(GA.SelectionType.RANDOM, GA.InitialSolutionType.RANDOM, GA.CrossoverType.UNIFORM_ORDER_BASED, GA.MutationType.INVERSION_MUTATION),
#     "2": Configuration(GA.SelectionType.RANDOM, GA.InitialSolutionType.RANDOM, GA.CrossoverType.PMX, GA.MutationType.RECIPROCAL_EXCHANGE),
#     "3": Configuration(GA.SelectionType.STOCHASTIC_UNIVERSAL_SAMPLING, GA.InitialSolutionType.RANDOM,
#                               GA.CrossoverType.UNIFORM_ORDER_BASED, GA.MutationType.RECIPROCAL_EXCHANGE),
#     "4": Configuration(GA.SelectionType.STOCHASTIC_UNIVERSAL_SAMPLING, GA.InitialSolutionType.RANDOM,
#                               GA.CrossoverType.PMX, GA.MutationType.RECIPROCAL_EXCHANGE),
#     "5": Configuration(GA.SelectionType.STOCHASTIC_UNIVERSAL_SAMPLING, GA.InitialSolutionType.RANDOM,
#                               GA.CrossoverType.PMX, GA.MutationType.INVERSION_MUTATION),
#     "6": Configuration(GA.SelectionType.STOCHASTIC_UNIVERSAL_SAMPLING, GA.InitialSolutionType.RANDOM,
#                               GA.CrossoverType.UNIFORM_ORDER_BASED, GA.MutationType.INVERSION_MUTATION),
#     "7": Configuration(GA.SelectionType.STOCHASTIC_UNIVERSAL_SAMPLING, GA.InitialSolutionType.HEURISTIC,
#                               GA.CrossoverType.PMX, GA.MutationType.RECIPROCAL_EXCHANGE),
#     "8": Configuration(GA.SelectionType.STOCHASTIC_UNIVERSAL_SAMPLING, GA.InitialSolutionType.HEURISTIC,
#                           GA.CrossoverType.UNIFORM_ORDER_BASED, GA.MutationType.INVERSION_MUTATION),
# })

configurations = dict({
        "1": Configuration(GA.SelectionType.RANDOM, GA.InitialSolutionType.RANDOM, GA.CrossoverType.UNIFORM_ORDER_BASED, GA.MutationType.INVERSION_MUTATION)

})

for filename in files:
    for population_size in population_sizes:
        for mutation_rate in mutation_rates:
            for test in range(0, number_of_test_iterations):
                print("File name: " + filename)
                print("Population size: " + str(population_size))
                print("Mutation rate: " + str(mutation_rate))
                print("Run: " + str(test + 1))
                print ("Total iterations: ", str(no_of_iterations))

                for key, config in configurations.items():
                    print(">>>>>>>>>>>>>>> Configuration " + key + " <<<<<<<<<<<<<<<<<<<<<<<")
                    ga = BasicTSP(filename, population_size, mutation_rate, no_of_iterations, config)
                    ga.search()
