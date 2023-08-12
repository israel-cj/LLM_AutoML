import copy
import numpy as np
import openai
from sklearn.model_selection import train_test_split
from .run_llm_code import run_llm_code
from data.similarity import TransferedPipelines


def get_prompt(
        name_dataset, hf_token, **kwargs
):
    similar_pipelines = TransferedPipelines(hf_token=hf_token, name_dataset=name_dataset, number_of_pipelines=5)
    return f"""
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
pipe.fit(X_train, y_train)

```end

Each codeblock generates exactly one useful pipeline. Which will be evaluated with Log loss. 
Remember, you 
Each codeblock ends with ```end and starts with "```python"
Codeblock:
"""

# Each codeblock either generates {how_many} or drops bad columns (Feature selection).


def build_prompt_from_df(name_dataset, hf_token):
    prompt = get_prompt(
        name_dataset,
        hf_token,
    )

    return prompt


def generate_features(
        X,
        y,
        model="gpt-3.5-turbo",
        just_print_prompt=False,
        iterative=1,
        iterative_method="logistic",
        display_method="markdown",
        n_splits=10,
        n_repeats=2,
        name_dataset = None,
        hf_token=None,
):
    def format_for_display(code):
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    if display_method == "markdown":
        from IPython.display import display, Markdown

        display_method = lambda x: display(Markdown(x))
    else:

        display_method = print
    prompt = build_prompt_from_df(name_dataset, hf_token, iterative=iterative)

    if just_print_prompt:
        code, prompt = None, prompt
        return code, prompt, None

    def generate_code(messages):
        if model == "skip":
            return ""

        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            stop=["```end"],
            temperature=0.5,
            max_tokens=500,
        )
        code = completion["choices"][0]["message"]["content"]
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    def execute_and_evaluate_code_block(code):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
        try:
            """We are here"""
            pipe = run_llm_code(
                code,
                X_train,
                y_train,
            )
        except Exception as e:
            display_method(f"Error in code execution. {type(e)} {e}")
            display_method(f"```python\n{format_for_display(code)}\n```\n")
            return e, None

        from contextlib import contextmanager
        import sys, os

        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                performance = pipe.score(X_test, y_test)
            finally:
                sys.stdout = old_stdout

        return None, performance

    messages = [
        {
            "role": "system",
            "content": "You are an expert datascientist assistant creating a Pipeline for a dataset X_train, y_train. You answer only by generating code. Answer as concisely as possible.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    display_method(f"*Dataset description:*\n {ds[-1]}")

    n_iter = iterative
    i = 0
    while i < n_iter:
        try:
            code = generate_code(messages)
        except Exception as e:
            display_method("Error in LLM API." + str(e))
            continue
        i = i + 1
        e, performance = execute_and_evaluate_code_block(code)
        if e is not None:
            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""Code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```\n Generate next pipeline (fixing error?):
                                ```python
                                """,
                },
            ]
            continue

        if isinstance(performance, float):
            valid_pipeline = True
            pipeline_sentence = f"The code was executed and generated a Pipeline ´pipe´ with score {performance}"
        else:
            valid_pipeline = False
            pipeline_sentence = "The last code did not generate a valid ´pipe´, it was discarded."

        display_method(
            "\n"
            + f"*Iteration {i}*\n"
            + f"*Valie pipeline {str(valid_pipeline)}*\n"
            + f"```python\n{format_for_display(code)}\n```\n"
            + f"Performance {performance} \n"
            + f"{pipeline_sentence}\n"
            + f"\n"
        )

    return code, prompt, messages