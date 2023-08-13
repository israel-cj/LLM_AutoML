# LLM_AutoML

This is a project based on CAAFE, semi-automate your feature engineering process.

The prompt will be something like this:

The dataframe split in ‘X_train’ and ‘y_train’ is loaded in memory.
This code was written by an expert data scientist working to create a suitable pipeline (preprocessing techniques and estimator) given such a dataset. It is a snippet of code that import the packages necessary to create a ‘sklearn’ pipeline together with a description. This code takes inspiration from previous similar pipelines and their respective ‘Log loss’ which worked for related ‘X_train’ and ‘y_train’. Those examples contain the word ‘Pipeline’ which refers to the preprocessing steps (optional) and estimators necessary, the word ‘data’ refers to ‘X_train’ and ‘y_train’ used during training, and finally ‘Log loss’ represent the performance of the model (the closes to 0 the better):
“
{similar_pipelines}
“

For instance, let’s consider you took inspiration from the next pipeline given that its ‘Log loss’ was the smallest from the examples provided above:

"Pipeline: GradientBoostingClassifier(RBFSampler(Normalizer(data, Normalizer.norm='l1'), RBFSampler.gamma=0.35000000000000003), GradientBoostingClassifier.learning_rate=0.5, GradientBoostingClassifier.max_depth=10, GradientBoostingClassifier.max_features=0.2, GradientBoostingClassifier.min_samples_leaf=10, GradientBoostingClassifier.min_samples_split=17, GradientBoostingClassifier.n_estimators=100, GradientBoostingClassifier.subsample=0.3) Log loss: 0.4515132104250264"

From the inspired snippet would expect something like the next codeblock:

````python
# Description: This pipeline is built using Gradient Boosting for classification. This algorithm builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. It is necessary to normalize the data before feeding the model

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import Normalizer
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline

step_1 = ('Normalizer', Normalizer(norm='l1'))
step_2 = ('RBFSampler', RBFSampler(gamma=0.38))
step_3 = ('GradientBoostingClassifier', GradientBoostingClassifier(n_estimators=90,
                                                                learning_rate=0.4,
                                                                max_depth=11,
                                                                min_samples_split=18,
                                                                min_samples_leaf=11,
                                                                subsample=0.3,
                                                                max_features=0.2))

pipe = Pipeline([step_1, step_2, step_3])
pipe = pipe.fit(X_train, y_train)

```end

Each codeblock generates exactly one useful pipeline. Which will be evaluated with Log loss.
Remember, you
Each codeblock ends with ```end and starts with "```python"
Codeblock:
````
