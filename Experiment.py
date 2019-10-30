import statistics
from datetime import datetime
from pathlib import Path
import pickle

class Experiment:

    DIR_NAME="results/"

    def __init__(self,  configuration, instance, no_of_iterations, mutation_rate, population_size):
        self.instance = instance
        self.noOfIterations = no_of_iterations
        self.mutationRate = mutation_rate
        self.populationSize = population_size
        self.mutation = configuration.mutation
        self.crossover = configuration.crossover
        self.selection = configuration.selection
        self.initialSolution = configuration.initial_solution
        self.testResults = {}
        data_folder = Path(self.DIR_NAME)
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        filename = str(timestamp) + ".txt"
        self.file_to_open = data_folder / filename
        f = open(self.file_to_open, "wb")
        pickle.dump(self, f)
        f.close()

    def saveResult(self, run, best_solution):
        f = open(self.file_to_open, 'ab')
        pickle.dump({run: best_solution}, f)
        f.close()

    def calculateBest(self):
        size = len(self.testResults)
        if size > 0:
            return min(individual.getFitness() for individual in self.testResults.values())
        return 0

    def calculateWorst(self):
        size = len(self.testResults)
        if size > 0:
            return max(individual.getFitness() for individual in self.testResults.values())
        return 0

    def calculateAverage(self):
        size = len(self.testResults)
        if size > 0:
            return sum(individual.getFitness() for individual in self.testResults.values())/len(self.testResults)
        return 0

    def calculateOptimal(self):
        size = len(self.testResults)
        if size > 0:
            return statistics.median(self.testResults.values())
        return 0

