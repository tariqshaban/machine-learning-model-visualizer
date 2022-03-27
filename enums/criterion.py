from enum import Enum


class Criterion(Enum):
    """
    Enumerate decision tree criterion
    """

    GINI = 'gini'

    ENTROPY = 'entropy'
