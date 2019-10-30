class Configuration:
    def __init__(self, _name, _selection, _initial_solution, _crossover, _mutation ):
        self.name = "Configuration " + _name
        self.selection =_selection
        self.initial_solution = _initial_solution
        self.crossover = _crossover
        self.mutation =_mutation

    def __str__(self):
        return self.name + "\nSelection: " + str(self.selection) \
               + "\nInitial Solution: " + str(self.initial_solution) \
               + "\nCrossover: " + str(self.crossover) \
               + "\nMutation: " + str(self.mutation)