"""
The `ez_data_quality` function processes a dataset to assess its quality based on various parameters and returns the results in a structured response. Here's a summary of what it does:

1. Parameter Validation:

   - It checks that required parameters (`train_data` and `outcome`) are provided, and if not, returns an error message.
   - It also validates the `options` argument (if provided) to ensure it's a dictionary and contains valid keys.

2. Configuration Setup:

   - It initializes configuration options, including handling specific keys related to data quality (e.g., `data_quality_options`, `prediction_data`).
   - If certain keys are invalid or have incorrect data types, it returns an error.

3. Data Processing:

   Based on the options specified (e.g., `data_shape`, `data_emptiness`, `remove_outliers`, `data_balance`, `outcome_correlation`), it performs various checks or transformations on the data:

   - `data_shape_quality`: Analyzes the shape of the data.
   - `data_emptiness_quality`: Checks for missing values and applies imputation if specified.
   - `data_outliers_quality`: Identifies and handles outliers.
   - `data_balance_quality`: Assesses the balance of the outcome variable.
   - `data_correlation_quality`: Analyzes the correlation of the data with the outcome variable.

4. Alert Generation:

   After evaluating the dataset, it generates quality alerts based on the results, flagging any issues related to the data.

5. Response:

   - It returns a structured response in JSON format indicating whether the data quality checks were successful or if there were any issues.
   - If an error occurs during processing, it returns an exception.
"""
import os
from flask import Response
from eazyml_insight import ez_init, ez_insight
import pandas as pd

from .globals import transparency_api as tr_api

from .src.utils import (
                    quality_alert_helper,
                    utility
)

from .src.main import (
    ez_correlation_local,
    ez_data_balance_local,
    ez_impute_local,
    ez_outlier_local,
    ez_shape_local,
    ez_drift_local,
    data_correctness_local,
    data_completeness_local

)

import json
from functools import partial

# from ...genai.eazyml_genai.genai.src.genai_utils.txt_utils import extract_json_content

convert_json = partial(json.dumps, indent=4, sort_keys=True, default=str)


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
        - **usage_share_consent** (`bool`): User's agreement to allow their usage information to be shared
    
    Returns:
        A dictionary containing the results of the initialization process with the following fields:
        
        - **success** (`bool`): Indicates whether the operation was successful.
        - **message** (`str`): A message describing the success or failure of the operation.

    Example:
        .. code-block:: python

            from eazyml import ez_init

            # Initialize the EazyML library with the access key.
            # This sets the `EAZYML_ACCESS_KEY` environment variable
            access_key = "your_access_key_here"  # Replace with your actual access key
            ez_init(access_key)

    Notes:
        - Make sure to call this function before using other functionalities of the EazyML library that require a valid access key.
        - The access key will be stored in the environment, and other functions in EazyML will automatically use it when required.
    """
    # update api and user info in hidden files
    init_resp = init_eazyml(access_key = access_key,
                                usage_share_consent=usage_share_consent,
                                usage_delete=usage_delete)
    return init_resp


def ez_data_quality(train_data, outcome, options = {}):
    """
    Performs a series of data quality checks on the given dataset and
    returns a JSON response indicating the results of these checks.

    Args:
        - **train_data** (`DataFrame` or `str`): A pandas DataFrame containing the training dataset. Alternatively, you can provide the file path of training dataset (as a string).
        - **outcome** (`str`): The target variable for the data quality.
        - **options** (`dict`, optional): A dictionary of options to configure the data quality process. If not provided, the function will use default settings. Supported keys include:
            
            - **data_shape** (`str`, optional): The default is `no`. If `yes`, the function will perform a data shape check.
            - **data_balance** (`str`, optional): The default is `no`. If `yes`, the function will perform a data balance check.
            - **data_emptiness** (`str`, optional): The default is `no`. If `yes`, the function will perform a data emptiness check.
            - **impute** (`str`, optional): The default is `no`. If `yes`, the function will perform imputation on training dataset.
            - **data_outliers** (`str`, optional): The default is `no`. If `yes`, the function will perform a data outliers check.
            - **remove_outliers** (`str`, optional): The default is `no`. If `yes`, the function will remove outliers from training dataset.
            - **outcome_correlation** (`str`, optional): The default is `no`. If `yes`, the function will perform a data correlation check.
            - **data_drift** (`str`, optional): The default is `no`. If `yes`, the function will perform a data drift check.
            - **model_drift** (`str`, optional): The default is `no`. If `yes`, the function will perform a model drift check.
            - **test_data** (`DataFrame` or `str`, optional): A pandas DataFrame containing the test dataset. Alternatively, you can provide the file path of test dataset (as a string).
            - **data_completeness** (`str`, optional): The default is `no`. If `yes`, the function will perform a data completeness check.
            - **dat_correctness** (`str`, optional): The default is `no`. If `yes`, the function will perform a data correctness check.

    Returns:
        - **dict**: A dictionary containing the results of the explanations with the following fields:
            
            - **success** (`bool`): Indicates whether the operation was successful.
            - **message** (`str`): A message describing the success or failure of the operation.

            
            **On Success**:
            
                - **data_shape_quality** (`dict`): Contains results of data shape quality checks.
                - **data_emptiness_quality** (`dict`): Includes results of data emptiness checks, such as the presence of missing or null values.
                - **data_outliers_quality** (`dict`): Provides insights into the presence of outliers.
                - **data_balance_quality** (`dict`): Contains information about the balance of data.
                - **data_correlation_quality** (`dict`): Includes results of correlation checks, identifying highly correlated features or potential redundancies.
                - **data_completeness_quality** (`dict`): Includes results of data completeness checks.
                - **data_correctness_quality** (`dict`): Includes results of data correctness checks.
                - **drift_quality** (`dict`): Includes results of data drift and model drift checks.
                - **data_bad_quality_alerts** (`dict`): Summarizes critical quality issues detected, with the following fields:
                - **data_shape_alert** (`bool`): Indicates if there are structural issues with the data (e.g., mismatched dimensions, irregular shapes).
                - **data_balance_alert** (`bool`): Flags issues with data balance (e.g., uneven class distributions).
                - **data_emptiness_alert** (`bool`): Signals significant levels of missing or null data.
                - **data_outliers_alert** (`bool`): Highlights the presence of extreme outliers that may affect data quality.
                - **data_correlation_alert** (`bool`): Flags excessive correlation among features that could lead to redundancy or multicollinearity.
                - **data_drift_alert** (`bool`): Flags data drift alerts based on ks data drift and psi data drift.
                - **model_drift_alert** (`bool`): Flags model drift alerts based on interval and distributional model drift.

    Example:
        .. code-block:: python
            
            from eazyml_data_quality import ez_data_quality

            # Define train data path (make sure the file path is correct).
            train_file_path = "path_to_your_train_data.csv"  # Replace with the correct file path

            # Define the outcome (target variable)
            outcome = "target"  # Replace with your actual target variable name

            # Define test data path (make sure the file path is correct).
            test_file_path = "path_to_your_test_data.csv"  # Replace with the correct file path

            # Set the options for data quality
            dqa_options = {
                           "data_shape": "yes",
                           "data_balance": "yes",
                           "data_emptiness": "yes",
                           "data_outliers": "yes",
                           "remove_outliers": "yes",
                           "outcome_correlation": "yes",
                           "data_drift": "yes",
                           "model_drift": "yes",
                           "prediction_data": test_file_path,
                           "data_completeness": "yes",
                           "data_correctness": "yes"
                          }

            # Call the eazyml function to perform data quality
            dqa_response = ez_data_quality(train_file_path, outcome, options=dqa_options)

            # dqa_response is a dictionary.
            dqa_response.keys()

            # Expected output (this will vary depending on the data):
            # dict_keys(['success', 'message', 'data_shape_quality', 'data_emptiness_quality', 'data_outliers_quality', 'data_balance_quality', 'data_correlation_quality', 'data_completeness_quality', 'data_correctness_quality', 'drift_quality', 'data_bad_quality_alerts'])

    """
    if not  outcome:
        return Response(response=convert_json(
            {
                "success": False,
                "message": tr_api.MANDATORY_PARAMETER % (["train_data", "outcome"]),
            }
        ),
            status=400,
            mimetype="application/json",
        )

    if options:
        ez_config = options
        if not isinstance(ez_config, dict):
            return Response(response=convert_json(
                {
                    "success": False,
                    "message": tr_api.VALID_DATATYPE_DICT.replace("this", "options"),
                }
            ),
                status=422,
                mimetype="application/json",
            )
    else:
        ez_config = {}
    # Check for valid keys in the options dict
    is_string = lambda x: isinstance(x, str)
    for key in ez_config:
        if key not in tr_api.EZ_DATA_QUALITY_OPTIONS_KEYS_LIST:
            return Response(response=convert_json(
                {
                    "success": False,
                    "message": tr_api.INVALID_KEY % (key),
                }
            ),
                status=422,
                mimetype="application/json",
            )
        if "data_quality_options" == key:
            if type(ez_config[key]) != type({}):
                return Response(
                    response=convert_json(
                        {"success": False, "message": tr_api.INVALID_DATATYPE_PARAMETER % (key)}
                    ),
                    status=422,
                    mimetype="application/json",
                )
            continue
        if "prediction_data" == key:
            continue
        if (not is_string(ez_config[key]) or not ez_config[key] in ["yes", "no"]):
            return Response(
                response=convert_json(
                    {"success": False, "message": tr_api.INVALID_DATATYPE_PARAMETER % (key)}
                ),
                status=422,
                mimetype="application/json",
            )

    if "data_quality_options" in ez_config:
        data_quality_options = ez_config["data_quality_options"]
    else:
        data_quality_options = {}

    for key in data_quality_options:
        if key not in tr_api.EZ_DATA_QUALITY_OPTIONS_OPTIONS_KEYS_LIST:
            return Response(response=convert_json(
                {
                    "success": False,
                    "message": tr_api.INVALID_KEY % (key),
                }
            ),
                status=422,
                mimetype="application/json",
            )
    if "impute" in ez_config and ez_config["impute"] == "yes":
        ez_load_options = {
            "outcome": outcome,
            "accelerate": "no",
            "impute": "no",
            "outlier": "no",
            "shuffle": "no"
        }
    else:
        ez_load_options = {
            "outcome": outcome,
            "accelerate": "yes",
            "impute": "no",
            "outlier": "no",
            "shuffle": "no"
        }
    if "data_load_options" in data_quality_options:
        tmp_options = data_quality_options["data_load_options"]
        for key in tmp_options:
            if key not in ["outcome"]:
                ez_load_options[key] = tmp_options[key]

    if isinstance(train_data, str):
        if not os.path.exists(train_data):
            return {
                "success": False,
                "message": "train_file_path does not exist."
            }
        train_data = utility.get_df(train_data)
    elif isinstance(train_data, pd.DataFrame):
        train_data = train_data
    else:
        return {
            "success": False,
            "message": 'train_data should be of either string or DataFrame'
        }
    cat_list, _ = utility.identify_column_types(train_data)
    columns = train_data.columns.to_list()
    for c in columns:
        if c in cat_list:
            train_data[c] = train_data[c].fillna("Null")
        else:
            train_data[c] = train_data[c].fillna(0)

    df = train_data
    try:
        final_resp = {}

        if "data_shape" in ez_config and ez_config["data_shape"] == "yes":
            json_resp, status_code = ez_shape_local(df)
            # print("status code", status_code)
            if status_code != 200:
                return Response(
                    response=json_resp,
                    status=status_code,
                    mimetype="application/json",
                )
            final_resp["data_shape_quality"] = json.loads(json_resp)
        # print('data_shape_quality', final_resp["data_shape_quality"])
        if "data_emptiness" in ez_config and ez_config["data_emptiness"] == "yes":
            impute_options = dict()
            if "impute" in ez_config:
                impute_options["impute"] = ez_config["impute"]
            # print('before impute')
            json_resp, status_code = ez_impute_local(df)
            # print('after impute', json_resp)
            if status_code != 200:
                return Response(
                    response=json_resp,
                    status=status_code,
                    mimetype="application/json",
                )
            final_resp["data_emptiness_quality"] = json.loads(json_resp)
        if "remove_outliers" in ez_config and ez_config["remove_outliers"] == "yes":
            outlier_options = dict()
            if "remove_outliers" in ez_config:
                outlier_options["remove_outliers"] = ez_config["remove_outliers"]
            json_resp, status_code = ez_outlier_local(df)
            if status_code != 200:
                return Response(
                    response=json_resp,
                    status=status_code,
                    mimetype="application/json",
                )
            final_resp["data_outliers_quality"] = json.loads(json_resp)
        # print('data_outliers_quality', final_resp["data_outliers_quality"])
        if "data_balance" in ez_config and ez_config["data_balance"] == "yes":
            json_resp, status_code = ez_data_balance_local(df, outcome)
            if status_code != 200:
                return Response(
                    response=json_resp,
                    status=status_code,
                    mimetype="application/json",
                )
            final_resp["data_balance_quality"] = json.loads(json_resp)
        # print('data_balance_quality', final_resp["data_balance_quality"])
        if "outcome_correlation" in ez_config and ez_config["outcome_correlation"] == "yes":
            json_resp, status_code = ez_correlation_local(df.copy(), outcome)
            if status_code != 200:
                return Response(
                    response=json_resp,
                    status=status_code,
                    mimetype="application/json",
                )
            final_resp["data_correlation_quality"] = json.loads(json_resp)

        res_augi = ez_insight(train_data, outcome, options={})
        if res_augi["success"] == True:
            insights_df = pd.DataFrame(res_augi['insights']['data'], columns=res_augi['insights']['columns'])
        else:
            return res_augi

        if "data_completeness" in ez_config and  ez_config["data_completeness"] == "yes":
            final_resp["data_completeness_quality"] = data_completeness_local(insights_df)

        if "data_correctness" in ez_config and ez_config["data_correctness"] == "yes":
            final_resp["data_correctness_quality"] = data_correctness_local(insights_df, outcome)



        drift_options = {}
        if "data_drift" in ez_config and ez_config["data_drift"] == "yes":
            drift_options["data_drift"] = ez_config["data_drift"]
        if "model_drift" in ez_config and ez_config["model_drift"] == "yes":
            drift_options["model_drift"] = ez_config["model_drift"]
        if drift_options != {}:
            if "prediction_data" not in ez_config:
                return Response(
                    response=convert_json(
                        {
                            "success": False,
                            "message": tr_api.MANDATORY_PARAMETER % (["prediction_data"]),
                        }
                    ),
                    status=400,
                    mimetype="application/json",
                )
            else:
                test_data = ez_config["prediction_data"]
            json_resp, status_code = ez_drift_local(test_data, train_data, outcome)
            if status_code != 200:
                return Response(
                    response=json_resp,
                    status=status_code,
                    mimetype="application/json",
                )
            final_resp["drift_quality"] = json.loads(json_resp)

        # print(final_resp["data_completeness_quality"])


        alerts = quality_alert_helper.quality_alerts(final_resp)
        final_resp["data_bad_quality_alerts"] = alerts

        final_resp["success"] = True
        final_resp["message"] = tr_api.DATA_QUALITY_SUCCESS
        return  final_resp

    except Exception as e:
        return e
