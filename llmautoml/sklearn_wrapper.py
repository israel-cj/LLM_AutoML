from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from .run_llm_code import run_llm_code
from .llmautoml import generate_features, list_pipelines
from .llmensemble import generate_code_embedding
from typing import Union
import numpy as np
import pandas as pd
import stopit
import uuid

class LLM_AutoML():
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
            make_ensemble = True,
            task="classification",
            max_total_time = 180,
    ) -> None:
        self.llm_model = llm_model
        self.iterations = iterations
        self.optimization_metric = optimization_metric
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.pipe = None
        self.make_ensemble = make_ensemble
        self.task = task
        self.timeout = max_total_time
        self.uid = str(uuid.uuid4())
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
        # global list_pipelines # To retrieve at least one result if the timeout is reached
        # Generate a unique UUID
        print('uid', self.uid)
        def get_score_pipeline(pipeline):
            # The split is only to make it faster
            if self.task == "classification":
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            performance = pipeline.score(X_train, y_train)
            return performance

        if self.task == "classification":
            y_ = y.squeeze() if isinstance(y, pd.DataFrame) else y
            self._label_encoder = LabelEncoder().fit(y_)
            if any(isinstance(yi, str) for yi in y_):
                # If target values are `str` we encode them or scikit-learn will complain.
                y = self._label_encoder.transform(y_)
                self._decoding = True

        # if self.task == 'regression':
        #     # Identify rows with missing values in X
        #     missing_rows = np.isnan(X).any(axis=1)
        #
        #     # Remove rows with missing values from X and y
        #     X = X[~missing_rows]
        #     y = y[~missing_rows]
        self.X_ = X
        self.y_ = y
        try:
            with stopit.ThreadingTimeout(self.timeout):
                self.code, prompt, messages, list_codeblocks_generated = generate_features(
                    X,
                    y,
                    model=self.llm_model,
                    iterative=self.iterations,
                    display_method="markdown",
                    n_splits=self.n_splits,
                    n_repeats=self.n_repeats,
                    task = self.task,
                    identifier=self.uid,
                )
            if len(list_pipelines)>0:
                self.pipe = list_pipelines[-1] # We get at least 1 pipeline
                self.pipe.fit(X, y)

            get_pipelines = []
            for code_pipe in list_codeblocks_generated:
                try:
                    this_pipe = run_llm_code(code_pipe, X, y)
                except Exception as e:
                    print(f"Exception: {e}")
                    this_pipe = None
                if isinstance(this_pipe, Pipeline):
                    get_pipelines.append(this_pipe)

            if len(get_pipelines) == 0:
                raise ValueError("Not pipeline could be created")

            if len(get_pipelines) == 1:
                self.pipe = get_pipelines[0]
            # Create an ensemble if we have more than 1 useful pipeline
            if len(get_pipelines) > 1 and self.make_ensemble:
                print('Creating an ensemble with LLM')
                self.pipe = generate_code_embedding(get_pipelines,
                                                    X,
                                                    y,
                                                    model=self.llm_model,
                                                    display_method="markdown",
                                                    task=self.task,
                                                    iterations_max=2,
                                                    identifier=self.uid,
                                                    )

                if self.pipe is None:
                    print('Ensemble with LLM failed, doing it manually')
                    if self.task == "classification":
                        from sklearn.ensemble import VotingClassifier
                        from sklearn.svm import SVC
                        from mlxtend.classifier import StackingClassifier

                        # Create the first layer of stackers
                        svc_rbf = SVC(kernel='rbf', probability=True)
                        stackers = []
                        for pipeline in get_pipelines:
                            stacker = StackingClassifier(classifiers=[pipeline],
                                                         meta_classifier=svc_rbf)
                            stackers.append(stacker)

                        # Create the second layer of stackers
                        estimators = [('stacker' + str(i), stacker) for i, stacker in enumerate(stackers)]
                        self.pipe = VotingClassifier(estimators=estimators, voting='soft')

                    else:
                        from sklearn.ensemble import VotingRegressor
                        # Create the ensemble
                        estimators = [('pipeline' + str(i), pipeline) for i, pipeline in enumerate(get_pipelines)]
                        self.pipe = VotingRegressor(estimators=estimators)

                # Fit the ensemble to the training data

            # Ensemble not allowed but more than one model in the list, the last model generated will be send it
            if len(get_pipelines) > 1 and self.make_ensemble == False:
                list_performance = [get_score_pipeline(final_pipeline) for final_pipeline in get_pipelines]
                # Index best pipeline:
                index_best_pipeline = list_performance.index(max(list_performance))
                # Return the one with the best performance
                print('Returning the best pipeline')
                self.pipe = get_pipelines[index_best_pipeline]

            self.pipe.fit(X, y)

        except stopit.TimeoutException:
            if self.pipe is None and len(list_pipelines)>0:
                self.pipe = list_pipelines[-1]
                self.pipe.fit(X, y)
            print("Timeout expired")

    def predict(self, X):
        if self.task == "classification":
            y = self.pipe.predict(X)  # type: ignore
            # Decode the predicted labels - necessary only if ensemble is not used.
            if y[0] not in list(self._label_encoder.classes_):
                y = self._label_encoder.inverse_transform(y)
            return y
        else:
            return self.pipe.predict(X)  # type: ignore

    def predict_log_proba(self, X):
        return self.pipe.predict_log_proba(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)






