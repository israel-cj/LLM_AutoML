
import openai
from sklearn.model_selection import train_test_split
from .run_llm_code import run_llm_code_ensemble


def get_prompt(task='classification'):
    return f"""
The dataframe split in ‘X_train’, ‘y_train’ and a list called ‘list_pipelines’ are loaded in memory.

This code was written by an expert data scientist working to create a suitable “Multi-Layer Stack Ensembling”, the task is {task}. It is a snippet of code that import the packages necessary to create a such Multi-layer stack ensembling model using a list of pipelines called ‘list_pipelines’, such list contain ‘sklearn’ pipeline objects. 

Code formatting the Multi-Layer Stack Ensembling:
```python
(Some packages imported and code necessary to create a Multi-Layer Stack Ensembling Model, which must be called ‘model’.
This model will be creating reusing all of its base layer model types “list_pipelines” as stackers. Those stacker models take as input not only the predictions of the models at the previous layer, but also the original data features themselves (input vectors are data features concatenated with lowerlayer model predictions).
The second and final stacking layer applies ensemble selection.
In addition, from 'model' call its respective 'fit' function to feed the model with 'X_train' and 'y_train')
```end

This codeblock ends with "```end" and starts with "```python"
Codeblock:

"""


# Each codeblock either generates {how_many} or drops bad columns (Feature selection).


def build_prompt_from(task='classification'):
    return get_prompt(task=task)

def generate_code_embedding(
        list_pipelines,
        X,
        y,
        model="gpt-3.5-turbo",
        display_method="markdown",
        task='classification',
    just_print_prompt=False,
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
    prompt = build_prompt_from(task=task)

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
            pipe = run_llm_code_ensemble(
                code,
                X_train,
                y_train,
                list_pipelines
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
            "content": f"You are an expert datascientist assistant creating a Multi-Layer Stack Ensembling for a dataset X_train, y_train, you need to use the pipelines storaged in 'list_pipelines’ . You answer only by generating code. Answer as concisely as possible. The task is {task}",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    display_method(f"*Dataset with specific description, task:*\n {task}")

    e = 1
    iteration_counts = 0
    while e is not None:
        iteration_counts+=1
        if iteration_counts>4:
            break
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
                    "content": f"""Code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```\n. Do it again and fix error.
                                ```python
                                """,
                },
            ]
            continue

        if isinstance(performance, float):
            valid_model = True
            pipeline_sentence = f"The code was executed and generated a Model ´model´ with score {performance}"
        else:
            valid_model = False
            pipeline_sentence = "The last code did not generate a valid ´model´, it was discarded."

        display_method(
            "\n"
            + f"*Valid model: {str(valid_model)}*\n"
            + f"```python\n{format_for_display(code)}\n```\n"
            + f"Performance {performance} \n"
            + f"{pipeline_sentence}\n"
            + f"\n"
        )

    return pipe