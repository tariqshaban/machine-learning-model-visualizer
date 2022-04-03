from enums.model import Model
from enums.criterion import Criterion
from enums.encoding import Encoding
from machine_learning_model_provider import MachineLearningModelProvider

########################################################################################################################
########################################################################################################################
# 1) Provide the directory in which the table is located
MachineLearningModelProvider.set_table_directory('assets/data.csv')

# 2) Specify the classification seed, if applicable
MachineLearningModelProvider.set_model_seed(43)

# 3) Specify the desired machine learning model, NOTE: may cause a runtime error if the columns were improper
MachineLearningModelProvider.set_selected_model(Model.DECISION_TREE_CLASSIFIER)

# 4) Specify the unwanted column(s) (if any)
MachineLearningModelProvider.set_unwanted_columns(['ID'])

# 5) Specify the table's label, this will inform the model of which column that should be treated as the label
MachineLearningModelProvider.set_label('Play')

# 6) Specify the table's label, this will inform the model of which column that should be treated as the label
MachineLearningModelProvider.set_train_test(test_size=0.3, seed=43)
########################################################################################################################
########################################################################################################################

# Get model's accuracy--------------------------------------------------------------------------------------------------
print(MachineLearningModelProvider.get_accuracy())

# Prints the decision tree on the console stream, the columns are label-encoded, tree is calculated using the Gini index
MachineLearningModelProvider.print_tree(criterion=Criterion.GINI, encoding=Encoding.LABEL_ENCODER)

# Plots the decision tree on the console stream, the columns are label-encoded, tree is calculated using the Gini index-
MachineLearningModelProvider.plot_tree(criterion=Criterion.GINI, encoding=Encoding.LABEL_ENCODER)
