from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from .run_llm_code import run_llm_code
from .llmpipeline import generate_features
import pandas as pd
import numpy as np
from typing import Optional
import pandas as pd


class LLM_pipeline():
    """
    A classifier that uses the CAAFE algorithm to generate features and a base classifier to make predictions.

    Parameters:
    base_classifier (object, optional): The base classifier to use. If None, a default TabPFNClassifier will be used. Defaults to None.
    optimization_metric (str, optional): The metric to optimize during feature generation. Can be 'accuracy' or 'auc'. Defaults to 'accuracy'.
    iterations (int, optional): The number of iterations to run the CAAFE algorithm. Defaults to 10.
    llm_model (str, optional): The LLM model to use for generating features. Defaults to 'gpt-3.5-turbo'.
    n_splits (int, optional): The number of cross-validation splits to use during feature generation. Defaults to 10.
    n_repeats (int, optional): The number of times to repeat the cross-validation during feature generation. Defaults to 2.
    """
    def __init__(
            self,
            optimization_metric: str = "accuracy",
            iterations: int = 10,
            llm_model: str = "gpt-3.5-turbo",
            n_splits: int = 10,
            n_repeats: int = 2,
            name_dataset = None,
            hf_token=None,
    ) -> None:
        self.llm_model = llm_model
        self.iterations = iterations
        self.optimization_metric = optimization_metric
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.name_dataset = name_dataset
        self.hf_token = hf_token
        self.pipe = None
    def fit(
            self, X, y, disable_caafe=False
    ):
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X : np.ndarray
            The training data features.
        y : np.ndarray
            The training data target values.

        """
        # if y.shape[1]>1:
        #    y =

        self.X_ = X
        self.y_ = y

        self.code, prompt, messages = generate_features(
            X,
            y,
            model=self.llm_model,
            iterative=self.iterations,
            iterative_method=self.base_classifier,
            display_method="markdown",
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            name_dataset = self.name_dataset,
            hf_token = self.hf_token
        )

        self.pipe = run_llm_code(
            self.code,
            X,
            y,
        )
        # Return the model
        return self.pipe

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_log_proba(self, X):
        return self.pipe.predict_log_proba(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)