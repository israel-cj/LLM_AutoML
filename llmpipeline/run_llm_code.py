import copy
import numpy as np
from typing import Any, Dict, Optional
import pandas as pd
import ast

def run_llm_code(code, X_train, y_train,):
    """
    Executes the given code on the given dataframe and returns the resulting dataframe.

    Parameters:
    code (str): The code to execute.
    df (pandas.DataFrame): The dataframe to execute the code on.
    convert_categorical_to_integer (bool, optional): Whether to convert categorical columns to integer values. Defaults to True.
    fill_na (bool, optional): Whether to fill NaN values in object columns with empty strings. Defaults to True.

    Returns:
    pandas.DataFrame: The resulting dataframe after executing the code.
    """
    try:
        access_scope = {"X_train": X_train, "y_train": y_train}
        parsed = ast.parse(code)
        pipe= exec(compile(parsed, filename="<ast>", mode="exec"), access_scope)

    except Exception as e:
        print("Code could not be executed", e)
        raise (e)

    return pipe


