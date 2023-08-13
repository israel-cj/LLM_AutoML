# LLM_AutoML

This is a project based on CAAFE, semi-automate your feature engineering process.

The prompt will be something like this:

The dataframe split in ‘X_train’ and ‘y_train’ is loaded in memory.
This code was written by an expert data scientist working to create a suitable pipeline (preprocessing techniques and estimator) given such a dataset. It is a snippet of code that import the packages necessary to create a ‘sklearn’ pipeline together with a description. This code takes inspiration from previous similar pipelines and their respective ‘Log loss’ which worked for related ‘X_train’ and ‘y_train’. Those examples contain the word ‘Pipeline’ which refers to the preprocessing steps (optional) and estimators necessary, the word ‘data’ refers to ‘X_train’ and ‘y_train’ used during training, and finally ‘Log loss’ represent the performance of the model (the closes to 0 the better):
“
{similar_pipelines}
“

Code formatting for each pipeline created:

````python
# Type of pipeline and description why this pipeline could work
(Some sklearn import packages and code using 'sklearn' to create a pipeline object 'pipe'. In addition call its respective 'fit' function to feed the model with 'X_train' and 'y_train')
```end

Each codeblock generates exactly one useful pipeline. Which will be evaluated with Log loss.
Each codeblock ends with "```end" and starts with "```python"
Make sure that along with the necessary preprocessing packages and sklearn models, always call 'Pipeline' from sklearn.
Codeblock:
"""

