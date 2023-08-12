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
dataset = openml.datasets.get_dataset(41078) # iris
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

df_train = pd.DataFrame(
        data=np.concatenate([X_train, np.expand_dims(y_train, -1)], -1), columns=attribute_names + [dataset.default_target_attribute]
    )
df_test = pd.DataFrame(
        data=np.concatenate([X_test, np.expand_dims(y_test, -1)], -1), columns=attribute_names + [dataset.default_target_attribute]
    )

### Setup and Run LLM pipeline - This will be billed to your OpenAI Account!

clf = LLM_pipeline(llm_model="gpt-3.5-turbo", iterations=3)

# The iterations happen here:
clf.fit_pandas(df_train,
                 target_column_name=dataset.default_target_attribute,
                 dataset_description=dataset.description)

# This process is done only once
pred = clf.predict(df_test)
acc = accuracy_score(pred, y_test)
print(f'LLM Pipeline accuracy {acc}')