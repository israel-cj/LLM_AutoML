import copy
import numpy as np
import openai
from sklearn.model_selection import train_test_split
from .run_llm_code import run_llm_code
from .similarity_description_based_mix import TransferedPipelines

def get_prompt(
        description_dataset=None, task='classification', number_of_pipelines= 3, **kwargs
):
    if task == 'classification':
        metric_prompt = 'Log loss'
    else:
        metric_prompt = 'Mean Squared Error'
    similar_pipelines = TransferedPipelines(description_dataset=description_dataset, task=task, number_of_pipelines=5)
    return f"""
The dataframe split in ‘X_train’ and ‘y_train’ is loaded in memory.

This code was written by an expert data scientist working to create a suitable 'Multi-Layer Stack Ensembling model', the task is {task}. This is done by creating {number_of_pipelines} not identical pipelines, a pipeline is the set of preprocessing techniques (if required) and estimator.
It is a snippet of code that import the packages necessary to create a ‘sklearn’ Multi-Layer Stack Ensembling model based on a list of pipelines.
To generate the list of pipelines with {number_of_pipelines} independent pipelines this codeblock takes inspiration from previous similar pipelines and their respective ‘{metric_prompt}’ which worked for related ‘X_train’ and ‘y_train’. 
The previous similar pipelines contain the word ‘Pipeline’ which refers to the preprocessing steps (optional) and estimators necessary, the word ‘data’ refers to ‘X_train’ and ‘y_train’ used during training, and finally ‘{metric_prompt}’ represent the performance of the model (the closest to 0 the better). The similar pipelines are:
“
{similar_pipelines}
“

Code formatting for the Multi-Layer Stack Ensembling model created:
```python
# Description of why this set of pipelines could work in the Multi-Layer Stack Ensembling
(
Some sklearn import packages and code using 'sklearn' to create {number_of_pipelines} pipelines with names 'pipe_1', 'pipe_2', ... 'pipe_{number_of_pipelines}'. Insert each pipe_X in a try-exception block, so if the code does not run for this one, still the next pipeline has a chance.
All pipelines running without errors should be storage in a list called 'list_pipelines' containing [pipe_1, ..., pipe_{number_of_pipelines}].
Then import the packages to create a Multi-Layer Stack Ensembling model by using 'list_pipelines', this model should be called 'stacking_model'
Finally, import the necessary packages to create the 'stacking_model', call its respective 'fit' function to feed it with 'X_train' and 'y_train'.
)
```end

This codeblock will be evaluated with "{metric_prompt}". 
This codeblock ends with "```end" and starts with "```python"
Codeblock:
"""

# Each codeblock either generates {how_many} or drops bad columns (Feature selection).


def build_prompt_from_df(description_dataset=None,
        task='classification'):

    prompt = get_prompt(
        description_dataset=description_dataset,
        task=task,
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
        description_dataset = None,
        task='classification'
):
    def format_for_display(code):
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    if display_method == "markdown":
        from IPython.display import display, Markdown

        display_method = lambda x: display(Markdown(x))
    else:

        display_method = print
    prompt = build_prompt_from_df(
        description_dataset=description_dataset,
        task=task,
    )

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
        if task == "classification":
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        try:
            pipe = run_llm_code(
                code,
                X_train,
                y_train,
            )
        except Exception as e:
            pipe = None
            display_method(f"Error in code execution. {type(e)} {e}")
            display_method(f"```python\n{format_for_display(code)}\n```\n")
            return e, None, None
        return None, pipe

    messages = [
        {
            "role": "system",
            "content": "You are an expert datascientist assistant creating a 'Multi-Layer Stack Ensembling model' for a dataset X_train, y_train. You answer only by generating code. Answer as concisely as possible.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    display_method(f"*Dataset with specific description, task:*\n {task}")

    try:
        code = generate_code(messages)
    except Exception as e:
        display_method("Error in LLM API." + str(e))
        pass
    e, pipe = execute_and_evaluate_code_block(code)

    if e is None:
        valid_pipeline = True
        pipeline_sentence = f"The code was executed and generated a Pipeline {pipe}"
    else:
        valid_pipeline = False
        pipeline_sentence = "The last code did not generate a valid ´pipe´, it was discarded."

    display_method(
        "\n"
        + f"*Valid pipeline: {str(valid_pipeline)}*\n"
        + f"```python\n{format_for_display(code)}\n```\n"
        + f"{pipeline_sentence}\n"
        + f"\n"
    )

    return code, prompt, messages