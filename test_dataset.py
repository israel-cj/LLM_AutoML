import openai
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from llmpipeline import LLM_pipeline
import openml

### Classification
# benchmark_ids = []
# suite = openml.study.get_suite(271) # Classification
# tasks = suite.tasks
# for task_id in tasks:
#     task = openml.tasks.get_task(task_id)
#     datasetID = task.dataset_id
#     benchmark_ids.append(datasetID)
#
# openai.api_key = ""
# type_task = "classification"
# # dataset = openml.datasets.get_dataset(41078) # iris
# dataset = openml.datasets.get_dataset(benchmark_ids[5]) # 5='wilt'
# X, y, categorical_indicator, attribute_names = dataset.get_data(
#     dataset_format="array", target=dataset.default_target_attribute
# )
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# generate_pipe = LLM_pipeline(llm_model="gpt-3.5-turbo", iterations=2, description_dataset=description_dataset, task=type_task)
#
# # The iterations happen here:
# clf = generate_pipe.fit(X_train, y_train)
#
# # This process is done only once
# y_pred = clf.predict(X_test)
# acc = accuracy_score(y_pred, y_test)
# print(f'LLM Pipeline accuracy {acc}')

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
description_dataset = dataset.description
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

### Setup and Run LLM pipeline - This will be billed to your OpenAI Account!
generate_pipe = LLM_pipeline(llm_model="gpt-3.5-turbo", iterations=2, description_dataset=description_dataset, task=type_task)

# The iterations happen here:
automl = generate_pipe.fit(X_train, y_train)

# This process is done only once
predictions = automl.predict(X_test)
print("LLM Pipeline MSE:", mean_squared_error(y_test, predictions))