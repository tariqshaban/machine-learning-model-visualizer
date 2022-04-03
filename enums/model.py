from enum import Enum


class Model(Enum):
    """
    Enumerate decision tree criterion
    """

    BAGGING_CLASSIFIER = 'BaggingClassifier'

    BAGGING_REGRESSION = 'BaggingRegression'

    SVC = 'SVC'

    LOGISTIC_REGRESSION = 'LogisticRegression'

    DECISION_TREE_CLASSIFIER = 'DecisionTreeClassifier'

    DECISION_TREE_REGRESSOR = 'DecisionTreeRegressor'

    RANDOM_FOREST_CLASSIFIER = 'RandomForestClassifier'

    RANDOM_FOREST_REGRESSOR = 'RandomForestRegressor'

    K_NEIGHBORS_CLASSIFIER = 'KNeighborsClassifier'

    K_NEIGHBORS_REGRESSOR = 'KNeighborsRegressor'

    K_MEANS = 'KMeansRegressor'

    LINEAR_REGRESSION = 'LinearRegression'

    XGB_CLASSIFIER = 'XGBClassifier'

    XGB_REGRESSOR = 'XGBRegressor'
