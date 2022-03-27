from enum import Enum


class Encoding(Enum):
    """
    Enumerate feature encoding types
    """

    LABEL_ENCODER = 'LabelEncoder'

    ONE_HOT_ENCODER = 'OneHotEncoder'
