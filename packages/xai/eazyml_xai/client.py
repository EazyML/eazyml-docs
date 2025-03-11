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

def ez_init(access_key: str=None,
                usage_share_consent: bool=None,
                usage_delete: bool=False):
    """
    Initialize EazyML package by passing `access_key`
    
    Args:
        - **access_key** (`str`): The access key to be set as an environment variable for EazyML.
        - **usage_share_consent** (`bool`): User's agreement to allow their usage information to be shared. If consent is given, only OS information, Python version, and EazyML packages API call counts are collected.
    
    Returns:
        A dictionary containing the results of the initialization process with the following fields:
        
        - **success** (`bool`): Indicates whether the operation was successful.
        - **message** (`str`): A message describing the success or failure of the operation.

    Example:
        .. code-block:: python

            from eazyml_xai import ez_init

            # Initialize the EazyML library with the access key.
            # This sets the `EAZYML_ACCESS_KEY` environment variable
            access_key = "your_access_key_here"  # Replace with your actual access key
            _ = ez_init(access_key)

    Notes:
        - Make sure to call this function before using other functionalities of the EazyML library that require a valid access key.
        - The access key will be stored in the environment, and other functions in EazyML will automatically use it when required.
    """
    # update api and user info in hidden files
    init_resp = init_eazyml(access_key = access_key,
                                usage_share_consent=usage_share_consent,
                                usage_delete=usage_delete)
    return init_resp


def ez_explain(train_data, outcome, test_data, model_info,
               options={}):
    """
    This API generates explanations for a model's prediction, based on provided train and test data.

    Args:
        - **train_data** (`DataFrame` or `str`): A pandas DataFrame containing the training dataset. Alternatively, you can provide the file path of training dataset (as a string).
        - **outcome** (`str`): The target variable for the explanations.
        - **test_data** (`DataFrame` or `str`): A pandas DataFrame containing the test dataset. Alternatively, you can provide the file path of test dataset (as a string).
        - **model_info** (`Bytes` or `object`): Contains the encrypted or unencrypted details about the trained model and its environment. Alternatively, you can provide the model trained on training dataset (as a object).
        - **options** (`dict`, optional): A dictionary of options to configure the explanation process. If not provided, the function will use default settings. Supported keys include:
            
            - **record_number** (`list`, optional): List of test data indices for which you want explanations. If not provided, it will compute the explanation for the first test data point.
            - **preprocessor** (`obj`, optional): Preprocessor that you used on the training dataset during preprocessing.

    Returns:
        A dictionary containing the results of the explanations with the following fields:
            
            - **success** (`bool`): Indicates whether the operation was successful.
            - **message** (`str`): A message describing the success or failure of the operation.

            **On Success**:
                - **explanations** (`dict`): The generated explanations contain the explanation string and local importance.

    Example Using EazyML Predictive Models:
        .. code-block:: python

            from eazyml_xai import ez_init, ez_explain

            # Initialize the EazyML automl library.
            _ = ez_init()

            # Load training data (Replace with the correct data path).
            train_data_path = "path_to_your_training_data.csv"
            train = pd.read_csv(train_data_path)

            # Load test data (Replace with the correct data path).
            test_data_path = "path_to_your_test_data.csv"
            test = pd.read_csv(test_data_path)

            # Define the outcome (target variable)
            outcome = "target"  # Replace with your target variable name

            # Build EazyML predictive models
            build_options = {'model_type': 'predictive'}
            build_resp = ez_build_model(train, outcome=outcome, options=build_options)

            # Use model_info from ez_build_model response
            model_info = build_resp["model_info"]

            # Customize options for fetching explanations
            xai_options = {"record_number": [1, 2, 3]}

            # Call the EazyML API to fetch the explanations
            xai_response = ez_explain(train, outcome, test_data_path, model_info, options=xai_options)

            # xai_response is a dictionary object with following keys.
            # print (xai_response.keys())
            # dict_keys(['success', 'message', 'explanations'])
       
    Example Using Your Model and EazyML Preprocessor:
        .. code-block:: python
            
            from eazyml_xai import ez_init, ez_explain, create_onehot_encoded_features, ez_get_data_type
            
            # Initialize the EazyML automl library.
            _ = ez_init()

            # Load training data (Replace with the correct data path).
            train_data_path = "path_to_your_training_data.csv"
            train = pd.read_csv(train_data_path)

            # Define the outcome (target variable)
            outcome = "target"  # Replace with your target variable name

            # Define input features (X) and target variable (y)
            y = train[outcome]
            X = train.drop(outcome, axis=1)

            # Load test data (Replace with the correct data path).
            test_data_path = "path_to_your_test_data.csv"
            test = pd.read_csv(test_data_path)

            # Get data type of features
            type_df = ez_get_data_type(train, outcome)

            # List of categorical columns
            cat_list = type_df[type_df['Data Type'] == 'categorical']['Variable Name'].tolist()
            cat_list = [ele for ele in cat_list if ele != outcome]

            # Create one-hot encoded features
            train = create_onehot_encoded_features(train, cat_list)

            # Define your model object (replace with any model of your choice)
            model_info = <YourModelClass>(<parameters>)  # e.g., RandomForestClassifier(), LogisticRegression(), etc.

            # Train your model object
            model_info.fit(X, y)

            # Customize options for fetching explanations
            xai_options = {"record_number": [1, 2, 3]}

            # Call the EazyML API to fetch the explanations
            xai_response = ez_explain(train, outcome, test_data_path, model_info, options=xai_options)

            # xai_response is a dictionary object with following keys.
            # print (xai_response.keys())

    Example Using Your Model and Preprocessor:
        .. code-block:: python

            from eazyml_xai import ez_init, ez_explain
            
            # Initialize the EazyML automl library.
            _ = ez_init()

            # Load training data (Replace with the correct data path).
            train_data_path = "path_to_your_training_data.csv"
            train = pd.read_csv(train_data_path)

            # Define the outcome (target variable)
            outcome = "target"  # Replace with your target variable name

            # Define input features (X) and target variable (y)
            y = train[outcome]
            X = train.drop(outcome, axis=1)

            # Load test data (Replace with the correct data path).
            test_data_path = "path_to_your_test_data.csv"
            test = pd.read_csv(test_data_path)

            # Implement your preprocessing steps within a custom preprocessor class and define it
            # (Replace <YourPreprocessorClass> with the specific preprocessor class you're using)
            preprocessor = <YourPreprocessorClass>(<parameters>)  # Example: StandardScaler(), CustomPreprocessor()

            # Fit the preprocessor on your dataset
            preprocessor.fit(X, y)

            # Define your model object (replace with any model of your choice)
            model_info = <YourModelClass>(<parameters>)  # e.g., RandomForestClassifier(), LogisticRegression(), etc.

            # Train your model object
            model_info.fit(X, y)

            # Customize options for fetching explanations
            xai_options = {"record_number": [1, 2, 3], "preprocessor", preprocessor}

            # Call the EazyML API to fetch the explanations
            xai_response = ez_explain(train_data_path, outcome, test_data_path, model_info, options=xai_options)

            # xai_response is a dictionary object with following keys.
            # print (xai_response.keys())
            # dict_keys(['success', 'message', 'explanations'])
   
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
                train_data, test_data, data_type_dict, outcome, selected_features_list)
            for col in selected_features_list:
                if col not in train_data.columns.tolist():
                    return {
                            "success": False,
                            "message": "Test dataset is not consistent with the training dataset"
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


def create_onehot_encoded_features(df, cols):
    """Convert categorical variables into dummy/one-hot encoded variables.
    
    This function takes a DataFrame and a list of column names, and returns a new DataFrame where 
    the specified columns are transformed into one-hot encoded (dummy) variables.

    Args:
        - **df** (`pd.DataFrame`): pandas dataframe for which dummy features are to be created
        - **cols** ('list'): List of categorical columns to be encoded
    
    Returns:
        A DataFrame with the specified columns replaced by their corresponding one-hot encoded dummy variables.
    
    Example:
        .. code-block:: python

            from eazyml_xai import ez_init, ez_get_data_type, create_onehot_encoded_features
            
            # Initialize the EazyML automl library.
            _ = ez_init()

            # Load data (Replace with the correct data path).
            data_path = "path_to_your_data.csv"
            data = pd.read_csv(data_path)

            # Define the outcome (target variable)
            outcome = "target"  # Replace with your target variable name

            # Get data type of features
            type_df = ez_get_data_type(data, outcome)

            # List of categorical columns
            cat_list = type_df[type_df['Data Type'] == 'categorical']['Variable Name'].tolist()
            cat_list = [ele for ele in cat_list if ele != outcome]

            # Create one-hot encoded features
            onehot_encoded_df = create_onehot_encoded_features(data, cat_list)

            # The onehot_encoded_df is a one-hot encoded DataFrame.
   
    """
    df = df.fillna("Null")
    return pd.get_dummies(
        df, columns=cols, dummy_na=False, prefix=cols, prefix_sep="_")


def ez_get_data_type(df, outcome):
    """
    Identifies if the columns are categorical or numeric and produces a DataFrame containing data types

    Args:
        - **df** (`pd.DataFrame`): pandas dataframe for which data types are to be identified.
        - **outcome** ('str'): Outcome variable name from the df
    
    Returns:
        A DataFrame with Variable Name and corresponding Data Type
    
    Example:
        .. code-block:: python

            from eazyml_xai import ez_init, ez_get_data_type
            
            # Initialize the EazyML automl library.
            _ = ez_init()

            # Load data (Replace with the correct data path).
            data_path = "path_to_your_data.csv"
            data = pd.read_csv(data_path)

            # Define the outcome (target variable)
            outcome = "target"  # Replace with your target variable name

            # Get data type of features
            type_df = ez_get_data_type(data, outcome)

            # type_df is a dataframe with following columns.
            # print (type_df.columns)
            # Index(['Variable Name', 'Data Type'], dtype='object')

    """
    df = df.fillna("Null")
    mode, data_type_dict, selected_features =\
        exai.get_mode_data_type_selected_features(df, outcome)
    type_df = pd.DataFrame(data_type_dict.items(), columns=[
        "Variable Name", "Data Type"])
    return type_df
