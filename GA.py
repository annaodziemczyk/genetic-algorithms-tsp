from enum import Enum

class CrossoverType(Enum):
    UNIFORM_ORDER_BASED=1,
    PMX=2

class InitialSolutionType(Enum):
    RANDOM=1,
    HEURISTIC=2

class MutationType(Enum):
    INVERSION_MUTATION=1,
    RECIPROCAL_EXCHANGE=2

class SelectionType(Enum):
    RANDOM=1,
    STOCHASTIC_UNIVERSAL_SAMPLING=2