from llmpipeline import LLM_pipeline
import openai
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import openml
import pandas as pd
import numpy as np

benchmark_ids = []
suite = openml.study.get_suite(271) # Classification
# suite = openml.study.get_suite(269) # Regression
tasks = suite.tasks
for task_id in tasks:
    task = openml.tasks.get_task(task_id)
    datasetID = task.dataset_id
    benchmark_ids.append(datasetID)

openai.api_key = " "
hf_token = ''
name_dataset = 'ada'
# dataset = openml.datasets.get_dataset(41078) # iris
dataset = openml.datasets.get_dataset(benchmark_ids[0]) # iris
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

### Setup and Run LLM pipeline - This will be billed to your OpenAI Account!

clf = LLM_pipeline(llm_model="gpt-3.5-turbo", iterations=3, name_dataset=name_dataset, hf_token=hf_token)

# The iterations happen here:
clf.fit(X_train, y_train)

# This process is done only once
pred = clf.predict(X_test)
acc = accuracy_score(pred, y_test)
print(f'LLM Pipeline accuracy {acc}')