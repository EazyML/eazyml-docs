import re
import os
import numpy as np
from .globals import transparency_api as tr_api
from .xai import exai
import traceback
import pandas as pd

from .license.license import (
        validate_license,
        init_eazyml
)

from .globals import logger as log
log.initlog()

def ez_init(access_key=None,
                usage_share_consent=True,
                usage_delete=False):
    """
    Initialize the EazyML library with a access key by setting the `EAZYML_ACCESS_KEY` environment variable.

    Parameters :
        - **access_key (str)**:
            The access key to be set as an environment variable for EazyML.

    Examples
    --------
    >>> init_ez("your_access_key_here")
    This sets the `EAZYML_ACCESS_KEY` environment variable to the provided access key.

    Notes
    -----
    Make sure to call this function before using other functionalities of the EazyML library that require a valid access key.
    """
    # update api and user info in hidden files
    approved, msg = init_eazyml(access_key = access_key,
                                usage_share_consent=usage_share_consent,
                                usage_delete=usage_delete)
    return {
            "success": approved,
            "message": msg
        }


def ez_explain(train_data, outcome, test_data, model_info,
               options={}):
    """
    This API generates explanations for a model's prediction, based on provided train and test data files.

    Parameters :
        - **train_data** (`str`): Path to the training file used to build the model.
        - **outcome** (`str`): The column in the dataset that you want to predict.
        - **test_data** (`str`): Path to the test file for predictions.
        - **model_info** (dict):
          A dictionary containing the trained model and associated model information (e.g., the model object 
          and any necessary pre-processing steps).
        - **options** (dict):
          A dictionary of configuration settings for counterfactual inference, which may include:
          
          .. code-block:: python

             options = {
                 "record_number": ["list of test data indices for which we want explaination"], if not provided it will compute explaination for all test data. 
                 "scaler": preprocessing that we need to apply on test data.
             }

    Returns :
        - **Dictionary with Fields**:
            - `success` (`bool`): Indicates if the explanation generation was successful.
            - `message` (`str`): Describes the success or failure of the operation.
            - `explanations` (`list, optional`): The generated explanations (if successful) contains the explanation string and a local importance dataframe.

        **On Success**:  
        A JSON response with
        
        .. code-block:: json

            {
                "success": true,
                "message": "Explanation generated successfully",
                "explanations": {
                    "explanation_string": "...",
                    "local_importance": { ".." : ".." }
                }
            }

        **On Failure**:  
        A JSON response with
        
        .. code-block:: json

            {
                "success": false,
                "message": "Error message"
            }

        **Raises Exception**:
            - Captures and logs unexpected errors, returning a failure message.
    
    Example:
        .. code-block:: python

            ez_explain(
                train_data='train.csv',
                outcome='target',
                test_data='test.csv',
                model_info=my_model,
                options={"record_number": [1, 2, 3],
                         "scaler": my_preprocessor}
            )
    """
    try:
        data_source = "system"
        if ("data_source" in options and options[
            "data_source"] == "parquet"):
            data_source = "parquet"


        if isinstance(train_data, str):
            if not os.path.exists(train_data):
                return {
                    "success": False,
                    "message": "train_file_path does not exist."
                }
            train_data, _ = exai.get_df(train_data, data_source=data_source)
            # print(train_data.columns)
        elif isinstance(train_data, pd.DataFrame):
            train_data = train_data.replace(r'^\s*$', np.nan, regex=True)
            # print(train_data.columns)

        else:
            return {
                "success": False,
                "message": 'train_data should be of either string or DataFrame'
            }

        train_data = train_data.fillna("Null")

        mode, data_type_dict, selected_features_list = exai.get_mode_data_type_selected_features(train_data, outcome)
        type_df = pd.DataFrame(data_type_dict.items(), columns=["Variable Name", "Data Type"])
        if type(model_info) == bytes:
            try:
                dic = exai.decrypt_dict(model_info)
                selected_features_list = dic["model_data"]["features_selected"]
                selected_features_list.append(outcome)

                if "model_name" in options:
                    model_name = options["model_name"]
                    list_model = [d["Model"] for d in dic["model_data"]["Consolidated Metrics"]]


                    if model_name in list_model:
                        model = dic["model_data"]["Consolidated Metrics"]["Models"]["Model" == model_name]["Models"]["model"]
                    else:
                        return {
                            "success": False,
                            "message": "Please provide a valid model name from encrypted bytes"
                        }


                else: model = dic["model_data"]["Consolidated Metrics"][0]["Models"]["model"]
            except Exception as e:
                return {
                    "success": False,
                    "message": "Please provide a valid encrypted model"
                }
        elif type(model_info) == dict:
            try:
                dic = model_info
                selected_features_list = dic["model_data"]["features_selected"]
                selected_features_list.append(outcome)

                if "model_name" in options:
                    model_name = options["model_name"]
                    list_model = [d["Model"] for d in dic["model_data"]["Consolidated Metrics"]]

                    if model_name in list_model:
                        model = dic["model_data"]["Consolidated Metrics"]["Models"]["Model" == model_name]["Models"]["model"]

                    else:
                        return {
                            "success": False,
                            "message": "Please provide a valid model name from encrypted bytes"
                        }

                else: model = dic["model_data"]["Consolidated Metrics"][0]["Models"]["model"]


            except Exception as e:
                return {
                    "success": False,
                    "message": "Please provide a valid model dict"
                }

        else:
            model = model_info
            if "selected_features" in options:
                 selected_features_list = options["selected_features"]
            else:
                selected_features_list = selected_features_list



            # model, scaler = exai.get_model_info(train_data, outcome, model_info, type_df, selected_features_list)




        if not isinstance(data_type_dict, dict):
            return {
                    "success": False,
                    "message": tr_api.VALID_DATATYPE_DICT.replace(
                        "this", "data_type"),
                    }

        if outcome not in data_type_dict.keys():
            return {
                    "success": False,
                    "message": "Outcome is not present in data_type"
                    }
        for col in set(data_type_dict.values()):
            if col not in ['numeric', 'categorical']:
                return {
                        "success": False,
                        "message": "Please provide valid type in data_type.('numeric'/'categorical')"
                        }

        if isinstance(test_data, str):
            if not os.path.exists(test_data):
                return {
                    "success": False,
                    "message": "test_file_path does not exist."
                }
            test_data, _ = exai.get_df(test_data, data_source=data_source)
        elif isinstance(test_data, pd.DataFrame):
            test_data = test_data
        else:
            return {
                "success": False,
                "message": 'test_data should be of either string or DataFrame'
            }
        test_data = test_data.fillna("Null")


        for col in data_type_dict.keys():
            if col not in train_data.columns:
                return {
                        "success": False,
                        "message": col + " is not present in training data columns"
                        }
        if len(data_type_dict.keys()) < 2:
            return {
                    "success": False,
                    "message": "Please provide data type for all columns (on which model is trained) in data_type."
                    }
        if outcome not in train_data.columns:
            return {
                    "success": False,
                    "message": "Outcome is not present in training data columns"
                    }
        if mode not in ['classification', 'regression']:
            return {
                    "success": False,
                    "message": "Please provide valid mode.('classification'/'regression')"
                    }
        if mode == 'regression' and (train_data[
            outcome].dtype == 'object' or test_data[
            outcome].dtype == 'object'):
            return {
                    "success": False,
                    "message": "The type of the outcome column is a string, so the mode should be classification."
                    }
        if mode == 'classification' and (pd.api.types.is_float_dtype(
            train_data[outcome]) or pd.api.types.is_float_dtype(
            test_data[outcome])):
            return {
                    "success": False,
                    "message": "The type of the outcome column is a float, so the mode should be regression."
                    }
        if not isinstance(options, dict):
            return {
                    "success": False,
                    "message": tr_api.VALID_DATATYPE_DICT.replace(
                        "this", "options"),
                    }

        #Check for valid keys in the options dict
        is_list = lambda x: type(x) == list
        is_string = lambda x: isinstance(x, str)
        if (
            not is_string(mode)
            or not is_string(outcome)
            # or not is_string(train_file_path)
            # or not is_string(test_file_path)
        ):
            return {
                        "success": False,
                        "message": tr_api.ALL_STR_PARAM
                    }
        if "scaler" in options:
            scaler = options["scaler"]
        else:
            scaler = None
        for key in options:
            if key not in tr_api.EZ_EXPLAIN_OPTIONS_KEYS_LIST:
                return {"success": False,
                        "message": tr_api.INVALID_KEY % (key)}

        if "record_number" in options and options["record_number"]:
            record_number = options["record_number"]

            if is_string(record_number):
                record_number = record_number.split(',')
                record_number = [item.strip() for item in record_number]
            if not is_list(record_number) and not is_string(
                record_number) and not isinstance(record_number, int):
                return {"success": False,
                        "message": "'record_number' in the 'options' parameter has either negative values or invalid data types."}
            elif is_list(record_number) and not all([(is_string(
                x) and x.isdigit()) or isinstance(
                x, int) for x in record_number]):
                return {"success": False,
                        "message": "'record_number' in the 'options' parameter has either negative values or invalid data types."}
            elif is_string(record_number) and not record_number.isdigit():
                return {"success": False,
                        "message": "'record_number' in the 'options' parameter has either negative values or invalid data types."}
            elif isinstance(record_number, int) and record_number < 0:
                return {"success": False,
                        "message": "'record_number' in the 'options' parameter has either negative values or invalid data types."}
            elif is_list(record_number) and any([isinstance(
                x, int) and x < 0 for x in record_number]):
                return {"success": False,
                        "message": "'record_number' in the 'options' parameter has either negative values or invalid data types."}

            if is_list(record_number):
                rec_n = exai.get_records_list(record_number)
                if rec_n != -1:
                    record_number = rec_n
                else:
                    return {"success": False,
                            "message": "'record_number' in the 'options' parameter has either negative values or invalid data types."}

            if is_list(record_number):
                record_number = record_number
            elif isinstance(record_number, int):
                record_number = [str(record_number)]
            else:
                record_number = [record_number]
            test_data_rows_count = test_data.shape[0]
            for rec_number in record_number:
                if int(rec_number) > test_data_rows_count:
                    return {
                            "success": False,
                            "message": "'record_number' in the 'options' parameter has values more than number of rows in the prediction dataset."
                            }
        else:
            record_number = [1]
        if "preprocessor" in options and options["preprocessor"]:
            try:
                train_data, test_data, rule_lime_dict, cat_list =\
                    exai.preprocessor_steps(
                    options['preprocessor'], train_data, test_data,
                    data_type_dict, outcome)
            except Exception as e:
                return {
                        "success": False,
                        "message": "Please provide a valid trained preprocessor."
                       }
        else:
            train_data, test_data, global_info_dict, cat_list =\
                exai.preprocessing_steps(
                train_data, test_data, data_type_dict, outcome)
            for col in selected_features_list:
                if col not in train_data.columns.tolist():
                    return {
                            "success": False,
                            "message": "Please provide valid column name in selected_features_list"
                            }

            train_data, test_data, rule_lime_dict = exai.processing_steps(
                train_data, test_data, global_info_dict, selected_features_list)


        body = dict(
                train_data = train_data,
                test_data = test_data,
                outcome = outcome,
                criterion = mode,
                scaler = scaler,
                model = model,
                rule_lime_dict = rule_lime_dict,
                cat_list = cat_list,
                record_numbers = record_number
            )
        results = exai.get_explainable_ai(body)

        if results == 'model is not correct':
            return {
                        "success": False,
                        "message": "Please provide a valid trained model."
                    }
        elif results == 'scaler is not correct':
            return {
                        "success": False,
                        "message": "Please provide a valid trained scaler."
                    }
        if type(results) != list:
            return {
                        "success": False,
                        "message": tr_api.EXPLANATION_FAILURE
                    }
        return {
                    "success": True,
                    "message": tr_api.EXPLANATION_SUCCESS,
                    "explanations": results,
                }

    except Exception as e:
        log.log_db(traceback.print_exc())
        return {"success": False, "message": tr_api.INTERNAL_SERVER_ERROR}


def ez_create_dummy_features(df, cols):
    df = df.fillna("Null")

    return pd.get_dummies(df, columns=cols, dummy_na=False, prefix=cols, prefix_sep="_")

def ez_get_data_type(df, outcome):
    df = df.fillna("Null")
    mode, data_type_dict, selected_features = exai.get_mode_data_type_selected_features(df, outcome)

    type_df = pd.DataFrame(data_type_dict.items(), columns=["Variable Name", "Data Type"])

    return type_df

def ez_create_selected_features(df, outcome):
    df = df.fillna("Null")

    mode, data_type_dict, selected_features = exai.get_mode_data_type_selected_features(df, outcome)

    return  selected_features




