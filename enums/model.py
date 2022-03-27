from enum import Enum


class Model(Enum):
    """
    Enumerate decision tree criterion
    """

    SVC = 'SVC'

    LOGISTIC_REGRESSION = 'LogisticRegression'

    DECISION_TREE_CLASSIFIER = 'DecisionTreeClassifier'

    RANDOM_FOREST_REGRESSOR = 'RandomForestRegressor'

    K_NEIGHBORS_CLASSIFIER = 'KNeighborsClassifier'

    LINEAR_REGRESSION = 'LinearRegression'

    XGB_CLASSIFIER = 'XGBClassifier'

    XGB_REGRESSOR = 'XGBRegressor'
