from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from .run_llm_code import run_llm_code
from .llmautoml import generate_features, list_pipelines
from .llmensemble import generate_code_embedding, generate_ensemble_manually
from typing import Union
import numpy as np
import pandas as pd
import stopit
import uuid

class LLM_AutoML():
    """
    Parameters:
    """
    def __init__(
            self,
            iterations: int = 10,
            llm_model: str = "gpt-3.5-turbo",
            n_splits: int = 10,
            n_repeats: int = 2,
            do_stacking = True,
            stacking_manually = False,
            task="classification",
            max_total_time = 180,
    ) -> None:
        if not do_stacking:
            stacking_manually = False
        self.llm_model = llm_model
        self.iterations = iterations
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.pipe = None
        self.do_stacking = do_stacking
        self.stacking_manually = stacking_manually
        self.task = task
        self.timeout = max_total_time
        self.uid = str(uuid.uuid4())
        self.base_models = None
        self.manually_success = False
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
            # A small sample if the dataset is too large
            value_to_consider_for_fast_training = 5000
            if self.task == "classification":
                if len(X) >= value_to_consider_for_fast_training:
                    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                        train_size=value_to_consider_for_fast_training,
                                                                        stratify=y, random_state=0)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, stratify=y,
                                                                        random_state=0)
            else:
                if len(X) >= value_to_consider_for_fast_training:
                    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                        train_size=value_to_consider_for_fast_training,
                                                                        random_state=0)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, random_state=0)

            performance = pipeline.score(X_train, y_train)
            return performance

        if self.task == "classification":
            y_ = y.squeeze() if isinstance(y, pd.DataFrame) else y
            self._label_encoder = LabelEncoder().fit(y_)
            if any(isinstance(yi, str) for yi in y_):
                # If target values are `str` we encode them or scikit-learn will complain.
                y = self._label_encoder.transform(y_)
                self._decoding = True

        self.X_ = X
        self.y_ = y
        try:
            with stopit.ThreadingTimeout(self.timeout):
                self.code, prompt, messages, list_codeblocks_generated, list_performance_pipelines = generate_features(
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
                index_best_pipeline = list_performance_pipelines.index(max(list_performance_pipelines))
                self.pipe = list_pipelines[index_best_pipeline] # We get at least 1 pipeline to return
                # self.pipe.fit(X, y)

            get_pipelines = list_pipelines
            self.base_models = list_pipelines
            if len(get_pipelines) == 0:
                raise ValueError("Not pipeline could be created")

            if len(get_pipelines) == 1:
                self.pipe = get_pipelines[0]
            # Create an ensemble if we have more than 1 useful pipeline
            if len(get_pipelines) > 1 and self.do_stacking and not self.stacking_manually:
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

            if (self.pipe is None) or self.stacking_manually:
                self.pipe = generate_ensemble_manually(X, y, get_pipelines, task=self.task)
                self.manually_success = True
                # Early exit
                return


            # Fit the ensemble to the training data

            # Ensemble not allowed but more than one model in the list, the last model generated will be send it
            if len(get_pipelines) > 1 and self.do_stacking == False:
                print('Returning the best pipeline without ensemble')
                # list_performance = [get_score_pipeline(final_pipeline) for final_pipeline in get_pipelines]
                # Index best pipeline:
                index_best_pipeline = list_performance_pipelines.index(max(list_performance_pipelines))
                # Return the one with the best performance
                self.pipe = get_pipelines[index_best_pipeline]
            print('The model has been created, final fit')
            self.pipe.fit(X, y)

        except stopit.TimeoutException:
            print("Timeout expired")

            # Train in a small portion to return something basic at least
            # A small sample if the dataset is too large
            value_to_consider_for_fast_training = 5000
            if self.task == "classification":
                if len(X) >= value_to_consider_for_fast_training:
                    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                        train_size=value_to_consider_for_fast_training,
                                                                        stratify=y, random_state=0)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, stratify=y,
                                                                        random_state=0)
            else:
                if len(X) >= value_to_consider_for_fast_training:
                    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                        train_size=value_to_consider_for_fast_training,
                                                                        random_state=0)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, random_state=0)

            if self.pipe is None and len(list_pipelines)>0:
                index_best_pipeline = list_performance_pipelines.index(max(list_performance_pipelines))
                self.pipe = list_pipelines[index_best_pipeline]
                self.pipe.fit(X_train, y_train)


    def predict(self, X):
        # This step is to conver the data with the preprocessing step since stacking don't consider such steps
        if self.manually_success:  # Only applicable if the model was ensembled manually
            preprocessing_steps = list(self.base_models[0].named_steps.values())[:-1]
            numeric_X = preprocessing_steps[0].fit_transform(X)
            X = pd.DataFrame(numeric_X, columns=[f"{i}" for i in range(numeric_X.shape[1])])
        if self.task == "classification":
            y = self.pipe.predict(X)  # type: ignore
            # Decode the predicted labels - necessary only if ensemble is not used.
            if y[0] not in list(self._label_encoder.classes_):
                y = self._label_encoder.inverse_transform(y)
            return y
        else:
            return self.pipe.predict(X)  # type: ignore

    def predict_log_proba(self, X):
        # This step is to conver the data with the preprocessing step since stacking don't consider such steps
        if self.manually_success:  # Only applicable if the model was ensembled manually
            preprocessing_steps = list(self.base_models[0].named_steps.values())[:-1]
            numeric_X = preprocessing_steps[0].fit_transform(X)
            X = pd.DataFrame(numeric_X, columns=[f"{i}" for i in range(numeric_X.shape[1])])
        return self.pipe.predict_log_proba(X)

    def predict_proba(self, X):
        # This step is to conver the data with the preprocessing step since stacking don't consider such steps
        if self.manually_success: # Only applicable if the model was ensembled manually
            preprocessing_steps = list(self.base_models[0].named_steps.values())[:-1]
            numeric_X = preprocessing_steps[0].fit_transform(X)
            X = pd.DataFrame(numeric_X, columns=[f"{i}" for i in range(numeric_X.shape[1])])
        return self.pipe.predict_proba(X)






