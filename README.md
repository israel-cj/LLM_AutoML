# LLM_AutoML

This is a project based on CAAFE, semi-automate your feature engineering process. It is a tool that helps you to create a machine learning model with a few lines of code, where the entire pipeline is created with LLM.

There are two requirements, define if the problem is 'classification' or 'regression' and provide a the dataset split in 'X' and 'y'.


![LLM AutoML](animated_automl.gif)

## Classification

```python

import openai
import openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from llmautoml import LLM_AutoML

openai.api_key = " " # Introduce your OpenAI key (reminder, you can create a Key with a free account, up to €5 budget "21/08/2023", equivalent to approximately running this framework 500 or more with 3 pipelines solutions)

dataset = openml.datasets.get_dataset(40983) # 40983 is Wilt dataset: https://www.openml.org/search?type=data&status=active&id=40983
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

### Setup and Run LLM pipeline - This will be billed to your OpenAI Account!
automl = LLM_AutoML(
    llm_model="gpt-3.5-turbo", # You can choose "gpt-4" in case you in case you have a paid account
    iterations=3,
    max_total_time = 3600,
    )

automl.fit(X_train, y_train)

# This process is done only once
y_pred = automl.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print(f'LLM Pipeline accuracy {acc}')

```

## Regression

```python
import openai
import openml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from llmautoml import LLM_AutoML


openai.api_key = " " # Introduce your OpenAI key (reminder, you can create a Key with a free account, up to €5 budget "21/08/2023", equivalent to approximately running this framework 500 or more times with 3 pipelines solutions)
type_task = 'regression'
dataset = openml.datasets.get_dataset(41021) # 41021 is Moneyball dataset: https://www.openml.org/search?type=data&status=active&id=41021
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

### Setup and Run LLM pipeline - This will be billed to your OpenAI Account!
automl = LLM_AutoML(
    llm_model="gpt-3.5-turbo", # You can choose "gpt-4" in case you in case you have a paid account
    iterations=3,
    task=type_task,
    max_total_time=3600
    )

automl.fit(X_train, y_train)

# This process is done only once
y_pred = automl.predict(X_test)
print("LLM Pipeline MSE:", mean_squared_error(y_test, y_pred))

```
