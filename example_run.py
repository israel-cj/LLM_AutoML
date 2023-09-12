import openai
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from llmautoml import LLM_AutoML
import openml

# #
# openai.api_key = ""
# task_id = 2073
# task = openml.tasks.get_task(task_id)
# datasetID = task.dataset_id
# dataset_local = openml.datasets.get_dataset(datasetID) # Hard 41147
# print('dataset_local name', dataset_local.name)
# iterations = 3
#
# X, y, categorical_indicator, attribute_names = dataset_local.get_data(
#             dataset_format="dataframe", target=dataset_local.default_target_attribute
#         )
#
# type_task = "classification"
# dict_params = {
#     'task': type_task,
#     'llm_model': "gpt-3.5-turbo",
#     'iterations': iterations,
#     'make_ensemble': True,
#     'max_total_time': 60,
# }
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
#
# automl = LLM_AutoML(**dict_params)
# automl.fit(X_train, y_train)
#
# # This process is done only once
# y_pred = automl.predict(X_test)
# acc = accuracy_score(y_pred, y_test)
# print(f'LLM Pipeline accuracy {acc}')
# #probabilitties
# probabilities = automl.predict_proba(X_test)
#
# ## Classification
# benchmark_ids = []
# suite = openml.study.get_suite(271) # Classification
# tasks = suite.tasks
# for task_id in tasks:
#     task = openml.tasks.get_task(task_id)
#     datasetID = task.dataset_id
#     benchmark_ids.append(datasetID)
#
#
#### Regression
benchmark_ids = []
suite = openml.study.get_suite(269) # Regression
tasks = suite.tasks
for task_id in tasks:
    task = openml.tasks.get_task(task_id)
    datasetID = task.dataset_id
    benchmark_ids.append(datasetID)

openai.api_key = ""
type_task = 'regression'
dataset = openml.datasets.get_dataset(benchmark_ids[0]) # 0 is equal to moneyball
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataset", target=dataset.default_target_attribute
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

### Setup and Run LLM AutoML - This will be billed to your OpenAI Account!
automl = LLM_AutoML(
    llm_model="gpt-3.5-turbo",
    iterations=4,
    task=type_task,
    max_total_time=3600,
    make_ensemble= True)

# The iterations happen here:
automl.fit(X_train, y_train)

# This process is done only once
predictions = automl.predict(X_test)
print("LLM Pipeline MSE:", mean_squared_error(y_test, predictions))
