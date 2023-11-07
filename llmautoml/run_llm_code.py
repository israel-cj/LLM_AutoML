import copy
import numpy as np
from typing import Any, Dict, Optional
import pandas as pd
import ast

def run_llm_code(code, X_train, y_train, pipe=None):
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

        globals_dict = {'X_train': X_train, 'y_train': y_train}
        output = {}
        exec(code, globals_dict, output)
        #output = {}
        #exec(code, None, output)
        # Use the resulting pipe object
        pipe = output['pipe']
        print(pipe)

    except Exception as e:
        print("Code could not be executed", e)
        raise (e)

    return pipe


def run_llm_code_ensemble(code, X_train, y_train, list_pipelines, model=None):
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

        globals_dict = {'X_train': X_train, 'y_train': y_train, 'list_pipelines':list_pipelines}
        output = {}
        exec(code, globals_dict, output)
        #output = {}
        #exec(code, None, output)
        # Use the resulting pipe object
        model = output['model']
        print(model)

    except Exception as e:
        print("Code could not be executed", e)
        raise (e)

    return model

def run_llm_optimize(code, X_train, y_train):
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

        globals_dict = {'X_train': X_train, 'y_train': y_train}
        output = {}
        exec(code, globals_dict, output)
        #output = {}
        #exec(code, None, output)
        # Use the resulting pipe object
        list_pipelines = output['list_pipelines']
        print('list_pipelines', list_pipelines)

    except Exception as e:
        list_pipelines = []
        print("Code could not be executed", e)
        raise (e)

    return list_pipelines