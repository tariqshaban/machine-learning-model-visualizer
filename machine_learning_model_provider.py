import collections

import joblib
import numpy as np
import pandas as pd
import plotly
from sklearn import tree, preprocessing, metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from enums.model import Model
from enums.criterion import Criterion
from enums.encoding import Encoding


class MachineLearningModelProvider:
    """
    Static methods which provides an abstract layer to conduct simple machine learning operations.

    Attributes
    ----------
        __table_directory       Specify the directory from where to read the assets from
        __data_raw              Acts as a cache for storing raw data input
        __split_data            Specify whether to split data on train dataset and test dataset
        __data_train            Stores the training dataset
        __data_test             Stores the testing dataset
        __model                 Acts as a cache for storing the model
        __selected_criterion    The desired criterion [Criterion.GINI, Criterion.ENTROPY]
        __selected_encoding     The desired categorical data encoding [Encoding.LABEL_ENCODER, Encoding.OneHotEncoder]
        __unwanted_columns      The columns that should be omitted from the model
        __label                 The table's label column name

    Methods
    -------
        set_table_directory(directory: str):
            Specify the directory from where to read the assets from.
        set_unwanted_columns(unwanted_columns: list):
            Specify the columns that should be omitted from the model.
        set_label(label: str):
            Specify the table's label column name.
        set_selected_model(preferred_model: Model):
            Specify the preferred model.
        set_model_seed(seed: int = None):
            Specify the seed.
        __get_categorical_columns() -> list:
            Returns the column names that contains categorical data types.
        __get_numerical_columns() -> list:
            Returns the column names that contains numerical data types.
        __get_feature_names() -> list:
            Returns a list of features
        __get_data_raw() -> pd.DataFrame:
            Loads raw txt table from __table_directory directory into a dataframe.
        get_data_raw() -> pd.DataFrame:
            Calls __get_data_raw() if __data_raw is None, otherwise,
            it retrieves __data_raw immediately.
        set_train_test(test_size: int = 0.3) -> list:
            Splits the raw data into a training dataset and testing dataset.
        __label_encode(df: pd.DataFrame) -> pd.DataFrame:
            Converts categorical values into numerical values,
            by replacing it with values starting from 0.
        __one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
            Converts categorical values into numerical values,
            by replacing it with values  between 0 and 1 on multiple columns.
        __get_model(criterion: Criterion = Criterion.GINI,
                    encoding: Encoding = Encoding.LABEL_ENCODER, clusters: int = 8, neighbors: int = 5):
            Performs the modeling process on the dataframe.
        get_model(criterion: Criterion = Criterion.GINI,
                    encoding: Encoding = Encoding.LABEL_ENCODER, clusters: int = 8, neighbors: int = 5):
            Calls __get_model() if __model is None, otherwise,
            it retrieves __model immediately.
        __print_legend():
            Displays a mapping between the values that has been converted from discrete to continues.
        print_tree(criterion: Criterion = Criterion.GINI,
                    encoding: Encoding = Encoding.LABEL_ENCODER) -> DecisionTreeClassifier:
            Displays the tree that the classifier has built in the console.
        plot_tree(criterion: Criterion = Criterion.GINI,
                    encoding: Encoding = Encoding.LABEL_ENCODER) -> DecisionTreeClassifier:
            Visualizes the tree that the classifier has built as a plot.
        predict(data: pd.DataFrame, criterion: Criterion = Criterion.GINI,
                    encoding: Encoding = Encoding.LABEL_ENCODER) -> list:
            Deploys the model to predict a new value.
        get_accuracy(criterion: Criterion = Criterion.GINI, encoding: Encoding = Encoding.LABEL_ENCODER) -> float:
            Return the accuracy on the given test data and labels.
        save_model():
            Saves the model locally for future prediction.
        plot_classification_roc_curve():
            Compute Receiver operating characteristic (ROC) for classification models.
        plot_regression_curve():
            Compute Receiver operating characteristic (ROC) for regression models.
    """

    __table_directory: str = 'assets/data.csv'
    __data_raw: pd.DataFrame = None
    __split_data: bool = False
    __data_train: pd.DataFrame = None
    __data_test: pd.DataFrame = None
    __model = None
    __model_seed: int = None
    __selected_criterion: Criterion = ''
    __selected_encoding: Encoding = ''
    __selected_model: Model = Model.DECISION_TREE_CLASSIFIER

    __unwanted_columns: list = []
    __label: str = ''

    @staticmethod
    def set_table_directory(directory: str):
        """
        Specify the directory from where to read the assets from.

        :param str directory: The desired table directory.
        """
        if directory != '':
            MachineLearningModelProvider.__table_directory = directory
            MachineLearningModelProvider.__unwanted_columns = []
            MachineLearningModelProvider.__label = ''
            MachineLearningModelProvider.__selected_model = Model.DECISION_TREE_CLASSIFIER
            MachineLearningModelProvider.__model = None
            MachineLearningModelProvider.__model_seed = None

    @staticmethod
    def set_unwanted_columns(unwanted_columns: list):
        """
        Specify the columns that should be omitted from the model.

        :param str unwanted_columns: The undesired columns.
        """
        if unwanted_columns:
            MachineLearningModelProvider.__unwanted_columns = unwanted_columns

    @staticmethod
    def set_label(label: str):
        """
        Specify the table's label column name.

        :param str label: The table's label.
        """
        if label != '':
            MachineLearningModelProvider.__label = label

    @staticmethod
    def set_selected_model(preferred_model: Model):
        """
        Specify the preferred model.

        :param Model preferred_model: Specify the preferred model.
        """

        if preferred_model not in [e for e in Model]:
            raise ValueError('Invalid model value')

        MachineLearningModelProvider.__model = None
        MachineLearningModelProvider.__selected_model = preferred_model

    @staticmethod
    def set_model_seed(seed: int = None):
        """
        Specify the seed.

        :param int seed: Controls the shuffling applied to the data before applying the split.
        """

        MachineLearningModelProvider.__model_seed = seed

    @staticmethod
    def __get_categorical_columns() -> list:
        """
        Returns the column names that contains categorical data types.

        :return list: List of the categorical columns.
        """

        data = MachineLearningModelProvider.get_data_raw()

        categorical_columns = list(set(data.columns) - set(MachineLearningModelProvider.__get_numerical_columns()))
        if MachineLearningModelProvider.__label in categorical_columns:
            categorical_columns.remove(MachineLearningModelProvider.__label)

        return categorical_columns

    @staticmethod
    def __get_numerical_columns() -> list:
        """
        Returns the column names that contains numerical data types.

        :return list: List of the numerical columns.
        """

        data = MachineLearningModelProvider.get_data_raw()

        # noinspection PyProtectedMember
        numerical_columns = list(data._get_numeric_data().columns)
        if MachineLearningModelProvider.__label in numerical_columns:
            numerical_columns.remove(MachineLearningModelProvider.__label)

        return numerical_columns

    @staticmethod
    def __get_feature_names() -> list:
        """
        Returns a list of features

        :return list: List of features
        """

        model: DecisionTreeClassifier = MachineLearningModelProvider.get_model(
            criterion=MachineLearningModelProvider.__selected_criterion,
            encoding=MachineLearningModelProvider.__selected_encoding
        )

        if MachineLearningModelProvider.__selected_model == Model.XGB_CLASSIFIER or \
                MachineLearningModelProvider.__selected_model == Model.XGB_REGRESSOR:
            features = model.get_booster().feature_names
        else:
            features = model.feature_names_in_.tolist()

        return features

    @staticmethod
    def __get_data_raw() -> pd.DataFrame:
        """
        Loads raw txt table from __table_directory directory into a dataframe.

        :return list: The table as a dataframe.
        """

        MachineLearningModelProvider.__data_raw = pd.read_csv(MachineLearningModelProvider.__table_directory)

        df = MachineLearningModelProvider.__data_raw

        for i in df.columns[df.isnull().any(axis=0)]:
            df[i].fillna(df[i].mean(), inplace=True)

        MachineLearningModelProvider.__data_raw \
            .drop(MachineLearningModelProvider.__unwanted_columns, axis=1, inplace=True)

        return MachineLearningModelProvider.__data_raw

    @staticmethod
    def get_data_raw() -> pd.DataFrame:
        """
        Calls __get_data_raw() if __data_raw is None, otherwise, it retrieves __data_raw immediately.

        :return list: The table as a dataframe.
        """

        if MachineLearningModelProvider.__data_raw is None:
            MachineLearningModelProvider.__get_data_raw()

        return MachineLearningModelProvider.__data_raw

    @staticmethod
    def set_train_test(test_size: float = 0.3, seed: int = None) -> list:
        """
        Splits the raw data into a training dataset and testing dataset.

        :param int test_size: The percentage of the training set in relation to the testing set.
        :param int seed: Controls the shuffling applied to the data before applying the split.
        :return list: The table as a dataframe.
        """

        data = MachineLearningModelProvider.get_data_raw()

        var_train, var_test, res_train, res_test = \
            train_test_split(
                data.loc[:, data.columns != MachineLearningModelProvider.__label],
                data[MachineLearningModelProvider.__label],
                test_size=test_size,
                random_state=seed
            )

        var_train[MachineLearningModelProvider.__label] = res_train
        var_test[MachineLearningModelProvider.__label] = res_test

        MachineLearningModelProvider.__data_train = var_train.reset_index(drop=True)
        MachineLearningModelProvider.__data_test = var_test.reset_index(drop=True)

        MachineLearningModelProvider.__split_data = True

        return [MachineLearningModelProvider.__data_train, MachineLearningModelProvider.__data_test]

    @staticmethod
    def __label_encode(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts categorical values into numerical values, by replacing it with values starting from 0.

        :param pd.DataFrame df: The required dataframe.
        :return pd.DataFrame: A dataframe whose categorical columns converted to numerical using the label encoding.
        """

        output = df.copy()

        le = preprocessing.LabelEncoder()
        for i in range(len(output.columns)):
            if output.columns[i] in MachineLearningModelProvider.__get_categorical_columns():
                output.iloc[:, i] = le.fit_transform(output.iloc[:, i])

        return output

    @staticmethod
    def __one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts categorical values into numerical values,
        by replacing it with values between 0 and 1 on multiple columns.

        :param pd.DataFrame df: The required dataframe.
        :return pd.DataFrame: A dataframe whose categorical columns converted to numerical using the one hot encoding.
        """

        output = df.copy()

        enc = OneHotEncoder(handle_unknown='ignore')
        for column_name in [x for x in output.columns if x in MachineLearningModelProvider.__get_categorical_columns()]:
            enc_df = pd.DataFrame(enc.fit_transform(output[[column_name]]).toarray())
            enc_df.columns = enc.get_feature_names_out()
            output = output.join(enc_df)
            output.drop(column_name, axis=1, inplace=True)

        return output

    @staticmethod
    def __get_model(criterion: Criterion = Criterion.GINI,
                    encoding: Encoding = Encoding.LABEL_ENCODER,
                    clusters: int = 8,
                    neighbors: int = 5):
        """
        Performs the modeling process on the dataframe.

        :param str criterion: Specify the desired criterion [Criterion.GINI, Criterion.ENTROPY].
        :param str encoding: Specify the desired categorical data encoding
                                [Encoding.LABEL_ENCODER, Encoding.OneHotEncoder].
        :param int clusters: Specify the number of clusters in a K-Means model.
        :param int neighbors: Specify the number of neighbors in a KNN model.
        :return: A fitted model.
        """

        if criterion not in [e for e in Criterion]:
            raise ValueError('Invalid criterion value')
        if encoding not in [e for e in Encoding]:
            raise ValueError('Invalid encoding value')

        if MachineLearningModelProvider.__split_data:
            data = MachineLearningModelProvider.__data_train
        else:
            data = MachineLearningModelProvider.get_data_raw()

        if encoding == Encoding.LABEL_ENCODER:
            data = MachineLearningModelProvider.__label_encode(data)
        elif encoding == Encoding.ONE_HOT_ENCODER:
            data = MachineLearningModelProvider.__one_hot_encode(data)

        if MachineLearningModelProvider.__selected_model == Model.BAGGING_CLASSIFIER:
            clf = BaggingClassifier(random_state=MachineLearningModelProvider.__model_seed)
        elif MachineLearningModelProvider.__selected_model == Model.BAGGING_REGRESSION:
            clf = BaggingRegressor(random_state=MachineLearningModelProvider.__model_seed)
        elif MachineLearningModelProvider.__selected_model == Model.SVC:
            clf = SVC(random_state=MachineLearningModelProvider.__model_seed)
        elif MachineLearningModelProvider.__selected_model == Model.LOGISTIC_REGRESSION:
            clf = LogisticRegression(random_state=MachineLearningModelProvider.__model_seed)
        elif MachineLearningModelProvider.__selected_model == Model.DECISION_TREE_CLASSIFIER:
            clf = DecisionTreeClassifier(criterion=criterion.value,
                                         random_state=MachineLearningModelProvider.__model_seed)
        elif MachineLearningModelProvider.__selected_model == Model.DECISION_TREE_REGRESSOR:
            clf = DecisionTreeRegressor()
        elif MachineLearningModelProvider.__selected_model == Model.RANDOM_FOREST_CLASSIFIER:
            clf = RandomForestClassifier(random_state=MachineLearningModelProvider.__model_seed)
        elif MachineLearningModelProvider.__selected_model == Model.RANDOM_FOREST_REGRESSOR:
            clf = RandomForestRegressor(random_state=MachineLearningModelProvider.__model_seed)
        elif MachineLearningModelProvider.__selected_model == Model.K_NEIGHBORS_CLASSIFIER:
            clf = KNeighborsClassifier(n_neighbors=neighbors)
        elif MachineLearningModelProvider.__selected_model == Model.K_NEIGHBORS_REGRESSOR:
            clf = KNeighborsRegressor(n_neighbors=neighbors)
        elif MachineLearningModelProvider.__selected_model == Model.K_MEANS:
            clf = KMeans(n_clusters=clusters, random_state=MachineLearningModelProvider.__model_seed)
        elif MachineLearningModelProvider.__selected_model == Model.LINEAR_REGRESSION:
            clf = LinearRegression()
        elif MachineLearningModelProvider.__selected_model == Model.XGB_CLASSIFIER:
            clf = xgb.XGBClassifier(random_state=MachineLearningModelProvider.__model_seed)
        else:
            clf = xgb.XGBRegressor(booster='gblinear')

        clf.fit(data.loc[:, ~data.columns.isin([MachineLearningModelProvider.__label])],
                data[MachineLearningModelProvider.__label])

        MachineLearningModelProvider.__model = clf
        MachineLearningModelProvider.__selected_criterion = criterion
        MachineLearningModelProvider.__selected_encoding = encoding

        return MachineLearningModelProvider.__model

    @staticmethod
    def get_model(criterion: Criterion = Criterion.GINI,
                  encoding: Encoding = Encoding.LABEL_ENCODER, clusters: int = 8, neighbors: int = 5):
        """
        Calls __get_model() if __model is None, otherwise, it retrieves __model immediately.

        :param str criterion: Specify the desired criterion [Criterion.GINI, Criterion.ENTROPY].
        :param str encoding: Specify the desired categorical data encoding
                                [Encoding.LABEL_ENCODER, Encoding.OneHotEncoder].
        :param int clusters: Specify the number of clusters in a K-Means model.
        :param int neighbors: Specify the number of neighbors in a KNN model.
        :return: A fitted model.
        """

        if MachineLearningModelProvider.__model is None or \
                MachineLearningModelProvider.__selected_criterion != criterion or \
                MachineLearningModelProvider.__selected_encoding != encoding:
            MachineLearningModelProvider.__get_model(criterion=criterion, encoding=encoding,
                                                     clusters=clusters, neighbors=neighbors)

        return MachineLearningModelProvider.__model

    @staticmethod
    def __print_legend():
        """
        Displays a mapping between the values that has been converted from discrete to continues.
        """

        data = MachineLearningModelProvider.get_data_raw().copy()
        data.drop(MachineLearningModelProvider.__get_numerical_columns(), axis=1, inplace=True)

        data_converted = data.copy()

        le = preprocessing.LabelEncoder()
        for i in range(len(data.columns)):
            data.iloc[:, i] = le.fit_transform(data.iloc[:, i])

        data_arr = []
        data_converted_arr = []

        for i in range(len(data.columns)):
            data_arr.append(data.iloc[:, i].unique())
            data_converted_arr.append(data_converted.iloc[:, i].unique())

        for i in range(len(data_arr)):
            dictionary = dict(zip(data_arr[i], data_converted_arr[i]))
            dictionary = collections.OrderedDict(sorted(dictionary.items()))
            for key in dictionary:
                print(f'{dictionary[key]}\t==>\t{key}')
            print('-------------')

    @staticmethod
    def print_tree(criterion: Criterion = Criterion.GINI, encoding: Encoding = Encoding.LABEL_ENCODER) \
            -> DecisionTreeClassifier:
        """
        Displays the tree that the classifier has built in the console.

        :param str criterion: Specify the desired criterion [Criterion.GINI, Criterion.ENTROPY].
        :param str encoding: Specify the desired categorical data encoding
                                [Encoding.LABEL_ENCODER, Encoding.OneHotEncoder].
        :return DecisionTreeClassifier: A classifier resulted from the decision tree process.
        """

        model = MachineLearningModelProvider.get_model(criterion=criterion, encoding=encoding)

        text_representation = \
            tree.export_text(model, feature_names=MachineLearningModelProvider.__get_feature_names())

        print(text_representation)

        return model

    @staticmethod
    def plot_tree(criterion: Criterion = Criterion.GINI, encoding: Encoding = Encoding.LABEL_ENCODER) \
            -> DecisionTreeClassifier:
        """
        Visualizes the tree that the classifier has built as a plot.

        :param str criterion: Specify the desired criterion [Criterion.GINI, Criterion.ENTROPY].
        :param str encoding: Specify the desired categorical data encoding
                                [Encoding.LABEL_ENCODER, Encoding.OneHotEncoder].
        :return DecisionTreeClassifier: A classifier resulted from the decision tree process.
        """

        model = MachineLearningModelProvider.get_model(criterion=criterion, encoding=encoding)

        plt.figure(figsize=(10, 8))

        tree.plot_tree(model,
                       feature_names=MachineLearningModelProvider.__get_feature_names(),
                       filled=True)

        MachineLearningModelProvider.__print_legend()

        plt.show()

        return model

    @staticmethod
    def predict(data: pd.DataFrame, criterion: Criterion = Criterion.GINI,
                encoding: Encoding = Encoding.LABEL_ENCODER) -> list:
        """
        Deploys the model to predict a new value.

        :param pd.DataFrame data: The label-less dataframe to predict.
        :param str criterion: Specify the desired criterion [Criterion.GINI, Criterion.ENTROPY].
        :param str encoding: Specify the desired categorical data encoding
                                [Encoding.LABEL_ENCODER, Encoding.OneHotEncoder].
        :return list: Array of the predicted class
        """

        data.columns = [x for x in MachineLearningModelProvider.get_data_raw().columns if
                        x != MachineLearningModelProvider.__label]

        model = MachineLearningModelProvider.get_model(criterion=criterion, encoding=encoding)

        if encoding == Encoding.LABEL_ENCODER:
            data = MachineLearningModelProvider.__label_encode(data)
        elif encoding == Encoding.ONE_HOT_ENCODER:
            data = MachineLearningModelProvider.__one_hot_encode(data)

        for column in MachineLearningModelProvider.__get_feature_names():
            if column not in data.columns:
                data[column] = 0

        for column in data.columns:
            if column not in MachineLearningModelProvider.__get_feature_names():
                del data[column]

        data = data[MachineLearningModelProvider.__get_feature_names()]

        return list(model.predict(data))

    @staticmethod
    def get_accuracy(criterion: Criterion = Criterion.GINI, encoding: Encoding = Encoding.LABEL_ENCODER) -> float:
        """
        Return the accuracy on the given test data and labels.

        :param str criterion: Specify the desired criterion [Criterion.GINI, Criterion.ENTROPY].
        :param str encoding: Specify the desired categorical data encoding
                                [Encoding.LABEL_ENCODER, Encoding.OneHotEncoder].
        :return float: Mean accuracy percentage.
        """

        if not MachineLearningModelProvider.__split_data:
            raise ValueError('Model was not split; specify the split first using \'set_train_test\' method.')

        model = MachineLearningModelProvider.get_model(criterion=criterion, encoding=encoding)

        data_test = MachineLearningModelProvider.__data_test

        if encoding == Encoding.LABEL_ENCODER:
            data_test = MachineLearningModelProvider.__label_encode(data_test)
        elif encoding == Encoding.ONE_HOT_ENCODER:
            data_test = MachineLearningModelProvider.__one_hot_encode(data_test)

            data = MachineLearningModelProvider.__one_hot_encode(MachineLearningModelProvider.__data_raw)
            for column in data.columns:
                if column not in data_test:
                    data_test[column] = 0
            data_test.columns = data.columns

        data_test_features = data_test.loc[:, data_test.columns != MachineLearningModelProvider.__label]
        data_test_label = data_test[MachineLearningModelProvider.__label]

        data_test_features = \
            data_test_features[
                data_test_features.columns.intersection(MachineLearningModelProvider.__get_feature_names())]

        score = model.score(data_test_features, data_test_label)

        return score

    @staticmethod
    def plot_classification_roc_curve():
        """
        Compute Receiver operating characteristic (ROC) for classification models.
        """

        model = MachineLearningModelProvider.get_model(
            criterion=MachineLearningModelProvider.__selected_criterion,
            encoding=MachineLearningModelProvider.__selected_encoding)

        data_test = MachineLearningModelProvider.__data_test

        if MachineLearningModelProvider.__selected_encoding == Encoding.LABEL_ENCODER:
            data_test = MachineLearningModelProvider.__label_encode(data_test)
        elif MachineLearningModelProvider.__selected_encoding == Encoding.ONE_HOT_ENCODER:
            data_test = MachineLearningModelProvider.__one_hot_encode(data_test)

            data = MachineLearningModelProvider.__one_hot_encode(MachineLearningModelProvider.__data_raw)
            for column in data.columns:
                if column not in data_test:
                    data_test[column] = 0
            data_test.columns = data.columns

        data_test_features = data_test.loc[:, data_test.columns != MachineLearningModelProvider.__label]
        data_test_label = data_test[MachineLearningModelProvider.__label]

        probs = model.predict_proba(data_test_features)
        predicts = probs[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(data_test_label, predicts)
        roc_auc = metrics.auc(fpr, tpr)

        params = {
            'legend.frameon': False,
            'text.color': 'w',
            'ytick.color': 'w',
            'xtick.color': 'w',
            'axes.labelcolor': 'w',
            'axes.edgecolor': 'w'
        }
        plt.rcParams.update(params)

        fig, ax = plt.subplots(nrows=1, ncols=1)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, color='#4caf50', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        ax = plt.gca()
        ax.set_facecolor('#303030')
        fig.patch.set_facecolor('#303030')

        plt.savefig('roc.png', dpi=500)

        plt.show()

    @staticmethod
    def plot_regression_curve():
        """
        Compute a scatter plot for regression models.
        """

        model = MachineLearningModelProvider. \
            get_model(criterion=MachineLearningModelProvider.__selected_criterion,
                      encoding=MachineLearningModelProvider.__selected_encoding)

        data_test = MachineLearningModelProvider.__data_test

        if MachineLearningModelProvider.__selected_encoding == Encoding.LABEL_ENCODER:
            data_test = MachineLearningModelProvider.__label_encode(data_test)
        elif MachineLearningModelProvider.__selected_encoding == Encoding.ONE_HOT_ENCODER:
            data_test = MachineLearningModelProvider.__one_hot_encode(data_test)

            data = MachineLearningModelProvider.__one_hot_encode(MachineLearningModelProvider.__data_raw)
            for column in data.columns:
                if column not in data_test:
                    data_test[column] = 0
            data_test.columns = data.columns

        data_test_features = data_test.loc[:, data_test.columns != MachineLearningModelProvider.__label]
        data_test_label = data_test[MachineLearningModelProvider.__label]

        y_pred = model.predict(data_test_features)

        print("MAE: {}".format(np.abs(data_test_label - y_pred).mean()))
        print("RMSE: {}".format(np.sqrt(((data_test_label - y_pred) ** 2).mean())))

        coefficients = model.coef_
        intercept = model.intercept_
        if hasattr(intercept, "__len__"):
            intercept = intercept[0]
        print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(intercept, coefficients[0], coefficients[1]))

        df = pd.DataFrame(data={'shots': data_test_features['shots'],
                                'shots_on_target': data_test_features['shots_on_target'],
                                'label': data_test_label})

        x_min, x_max = data_test_features['shots'].min(), data_test_features['shots'].max()
        y_min, y_max = data_test_features['shots_on_target'].min(), data_test_features['shots_on_target'].max()
        xrange = np.arange(x_min, x_max)
        yrange = np.arange(y_min, y_max)
        xx, yy = np.meshgrid(xrange, yrange)

        df1 = pd.DataFrame(data={'shots': xx.ravel(),
                                 'shots_on_target': yy.ravel()
                                 })
        prediction = model.predict(df1)
        prediction = prediction.reshape(xx.shape)

        df.rename({'shots': 'Shots', 'shots_on_target': 'Shots on Target', 'label': 'Goals'}, axis=1, inplace=True)

        fig = px.scatter_3d(df,
                            x='Shots',
                            y='Shots on Target',
                            z='Goals',
                            title='Regression Scatter Plot Characteristic', )
        fig.update_traces(marker=dict(size=5, color='#4caf50'))
        fig.add_traces(go.Surface(x=xrange, y=yrange, z=prediction, name='Trend Line'))

        fig.update_xaxes(showgrid=True, gridwidth=5, gridcolor='Black')
        fig.update_yaxes(showgrid=True, gridwidth=5, gridcolor='Black')

        fig.update_layout(
            paper_bgcolor='#303030',
            plot_bgcolor='#303030',
            font_color='white',
            title_font_color='white',
            legend_title_font_color='white',
            scene=dict(
                xaxis=dict(
                    backgroundcolor='#303030',
                    gridcolor='white',
                    zerolinecolor='white'
                ),
                yaxis=dict(
                    backgroundcolor='#303030',
                    gridcolor='white',
                    zerolinecolor='white'
                ),
                zaxis=dict(
                    backgroundcolor='#303030',
                    gridcolor='white',
                    zerolinecolor='white'
                ),
            )
        )

        with open("Output.txt", "w") as text_file:
            text_file.write(plotly.offline.plot(fig, include_plotlyjs=False, output_type='div'))
        fig.show()

    @staticmethod
    def save_model():
        """
        Saves the model locally for future prediction.
        """

        joblib.dump(MachineLearningModelProvider.__model, 'model.pkl')
