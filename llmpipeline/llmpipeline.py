import copy
import numpy as np
import openai
from sklearn.model_selection import train_test_split
from .run_llm_code import run_llm_code
from .similarity import TransferedPipelines


def get_prompt(
        name_dataset, hf_token, task, **kwargs
):
    additional_data = ''
    if task == 'classification':
        metric_prompt = 'Log loss'
    else:
        metric_prompt = 'Mean Squared Error'
        additional_data = f"Make sure to always use 'SimpleImputer' since ‘Nan’ values are not allowed in {task}, and call ‘f_regression’ if it will be used in the Pipeline"
    similar_pipelines = TransferedPipelines(hf_token=hf_token, name_dataset=name_dataset, task=task, number_of_pipelines=5)
    return f"""
The dataframe split in ‘X_train’ and ‘y_train’ is loaded in memory.
This code was written by an expert data scientist working to create a suitable pipeline (preprocessing techniques and estimator) given such a dataset, the task is {task}. It is a snippet of code that import the packages necessary to create a ‘sklearn’ pipeline together with a description. This code takes inspiration from previous similar pipelines and their respective ‘{metric_prompt}’ which worked for related ‘X_train’ and ‘y_train’. Those examples contain the word ‘Pipeline’ which refers to the preprocessing steps (optional) and estimators necessary, the word ‘data’ refers to ‘X_train’ and ‘y_train’ used during training, and finally ‘{metric_prompt}’ represent the performance of the model (the closes to 0 the better):
“
{similar_pipelines}
“

Code formatting for each pipeline created:
```python
# Type of pipeline and description why this pipeline could work 
(Some sklearn import packages and code using 'sklearn' to create a pipeline object 'pipe'. In addition call its respective 'fit' function to feed the model with 'X_train' and 'y_train')
```end

Each codeblock generates exactly one useful pipeline. Which will be evaluated with "{metric_prompt}". 
Each codeblock ends with "```end" and starts with "```python"
Make sure that along with the necessary preprocessing packages and sklearn models, always call 'Pipeline' from sklearn.
{additional_data}
Codeblock:
""", similar_pipelines

# Each codeblock either generates {how_many} or drops bad columns (Feature selection).


def build_prompt_from_df(name_dataset, hf_token, task):
    prompt, similar_pipelines = get_prompt(
        name_dataset,
        hf_token,
        task,
    )

    return prompt, similar_pipelines


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
        task='classification'
):
    list_codeblocks = []
    def format_for_display(code):
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    if display_method == "markdown":
        from IPython.display import display, Markdown

        display_method = lambda x: display(Markdown(x))
    else:

        display_method = print
    prompt, similar_pipelines = build_prompt_from_df(name_dataset, hf_token, task)

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
            """We are here"""
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

        from contextlib import contextmanager
        import sys, os

        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                # Works for both, regression and classification, I guess
                performance = pipe.score(X_test, y_test)
            finally:
                sys.stdout = old_stdout

        return None, performance, pipe

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
    display_method(f"*Task:*\n {name_dataset}")

    n_iter = iterative
    i = 0
    while i < n_iter:
        try:
            code = generate_code(messages)
            list_codeblocks.append(code)
        except Exception as e:
            display_method("Error in LLM API." + str(e))
            continue
        i = i + 1
        e, performance, pipe = execute_and_evaluate_code_block(code)
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
            + f"*Valid pipeline: {str(valid_pipeline)}*\n"
            + f"```python\n{format_for_display(code)}\n```\n"
            + f"Performance {performance} \n"
            + f"{pipeline_sentence}\n"
            + f"\n"
        )
        if task =='classification':
            next_add_information = ''
        if task =='regression':
            next_add_information = f"Use 'SimpleImputer' since ‘Nan’ values are not allowed in {task} tasks, and call ‘f_regression’ if it will be used in the Pipeline"

        if len(code) > 10:
            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""The pipeline {pipe} provided a score of {performance}.  
                    Again, here are the similar Pipelines: 
                    {similar_pipelines}
                    
                    Generate Pipelines that are diverse and not identical to previous iterations.
                    Along with the necessary preprocessing packages and sklearn models, always call 'Pipeline' from sklearn. {next_add_information}.
        Next codeblock:
        """,
                },
            ]

    return code, prompt, messages, list_codeblocks