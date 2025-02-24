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

def ez_init(license_key=None):
    """
    Initialize the EazyML library with a license key by setting the `EAZYML_LICENSE_KEY` environment variable.

    Parameters :
        - **license_key (str)**:
            The license key to be set as an environment variable for EazyML.

    Examples
    --------
    >>> init_ez("your_license_key_here")
    This sets the `EAZYML_LICENSE_KEY` environment variable to the provided license key.

    Notes
    -----
    Make sure to call this function before using other functionalities of the EazyML library that require a valid license key.
    """
    # update api and user info in hidden files
    approved, msg = init_eazyml(license_key = license_key)
    return {
            "success": approved,
            "message": msg
        }


def ez_data_quality(train_data, outcome, options = {}):
    """
    Performs a series of data quality checks on the given dataset and
    returns a JSON response indicating the results of these checks.

    Parameters:
        - **train_data** (str):
            The path to the file containing the dataset or a pandas Dataframe containing the train data.
        - **outcome** (str):
            The target variable (outcome) to assess data quality against.
        - **options** (dict, optional):
            A dictionary specifying additional configurations for data quality checks. 

    Returns :
         - **Dictionary with Fields**:
            - **success** (bool): Indicates whether the operation was successful.
            - **message** (str): Provides details about the outcome or an error message if the operation failed.
            - **data_shape_quality** (dict, optional): Contains results of data shape quality checks.
            - **data_emptiness_quality** (dict, optional): Includes results of data emptiness checks, such as the presence of missing or null values.
            - **data_outliers_quality** (dict, optional): Provides insights into the presence of outliers.
            - **data_balance_quality** (dict, optional): Contains information about the balance of data.
            - **data_correlation_quality** (dict, optional): Includes results of correlation checks, identifying highly correlated features or potential redundancies.
            - **data_completeness_quality** (dict, optional): Includes results of data completeness checks.
            - **data_correctness_quality** (dict, optional): Includes results of data correctness checks.
            - **drift_quality** (dict, optional): Includes results of data drift and model drift checks.
            - **data_bad_quality_alerts** (dict, optional): Summarizes critical quality issues detected, with the following fields:
                - **data_shape_alert** (bool): Indicates if there are structural issues with the data (e.g., mismatched dimensions, irregular shapes).
                - **data_balance_alert** (bool): Flags issues with data balance (e.g., uneven class distributions).
                - **data_emptiness_alert** (bool): Signals significant levels of missing or null data.
                - **data_outliers_alert** (bool): Highlights the presence of extreme outliers that may affect data quality.
                - **data_correlation_alert** (bool): Flags excessive correlation among features that could lead to redundancy or multicollinearity.
                - **data_drift_alert** (bool): Flags data drift alerts based on ks data drift and psi data drift.
                - **model_drift_alert** (bool): Flags model drift alerts based on interval and distributional model drift.




        **On Success** :
        A JSON response with

        .. code-block:: json

            {
                "success": true,
                "message": "Data quality checks according to given options have been calculated successfully",
                "data_shape_quality": {
                    "Dataset_dimension": [".."],
                    "alert": [".."],
                    "message": "No of columns in dataset is not adequate because the no of rows in the dataset is less than the no of columns",
                    "success": true
                },
                "data_emptiness_quality": {
                    "message": "There are no missing values present in the training data that was uploaded. Hence no records were imputed.",
                    "success": true
                },
                "data_outliers_quality": {
                    "message": "The following data points were removed as outliers.",
                    "outliers": {
                        "columns": [".."],
                        "indices": [".."]
                    },
                    "success": true
                },
                "data_balance_quality": {
                    "data_balance": {
                        "data_balance_analysis": {
                            "balance_score": [".."],
                            "data_balance": true,
                            "decision_threshold": [".."],
                            "quality_message": "Uploaded data is balanced because the balance score is greater than given threshold"
                        }
                    },
                    "message": "Data balance has been checked successfully",
                    "success": true
                },
                "data_correlation_quality": {
                    "data_correlation": "dict containing column wise correlations",
                    "data_correlation_alert": "true",
                    "message": "Correlation has been calculated successfully between all features and all features with outcome",
                    "success": true
                },
                "data_completeness_quality": {
                    "completeness": "true",
                    "decision_threshold": 0.6,
                    "insight_information": "The uploaded dataset is complete at confidence level of 0.6",
                    "top_insight": [".."],
                    "top_score": [".."]
                },
                "data_correctness_quality": {
                    "insights": [".."],
                    "message": "Please infer the correctness through logical inspection of the insights",
                    "quality_alert": "Please verify that the above rules are making sense or not. In case there are one or more rules which appear incorrect from an expert perspective, please double check your files for the variables, their correct values, in the offending rules"
                },
                "drift_quality": {
                    "data_drift_analysis": {"dict containing ks and psi data drift analysis"
                    },
                    "model_drift_analysis": {"dict containing distributional and interval model drift analysis"
                    },
                    "message": "Drift has been calculated successfully",
                    "success": true
                },
                "data_bad_quality_alerts": {
                    "data_shape_alert": "true/false",
                    "data_balance_alert": "true/false",
                    "data_emptiness_alert": "true/false",
                    "data_outliers_alert": "true/false",
                    "data_drift_alert": "true/false",
                    "model_drift_alert": "true/false",
                    "data_correctness_alert": "Please infer the correctness through logical inspection of the insights",
                    "data_completeness_alert": "true/false",
                    "data_correlation_alert": "true/false"
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
            - Captures and logs unexpected errors, returning a failure message with an internal server error indication.


    Example :
        .. code-block:: python

            ez_data_quality(
                train_data = 'train/file/path.csv',
                outcome = "outcome column"
                options = {
                    "data_shape": "yes",
                    "data_balance": "yes",
                    "data_emptiness": "yes",
                    "data_outliers": "yes",
                    "remove_outliers": "yes",
                    "outcome_correlation": "yes",
                    "data_drift": "yes",
                    "model_drift": "yes",
                    "prediction_data": test_data,
                    "data_completeness": "yes",
                    "data_correctness": "yes"
                }
            )
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
            json_resp, status_code = ez_correlation_local(df, outcome)
            if status_code != 200:
                return Response(
                    response=json_resp,
                    status=status_code,
                    mimetype="application/json",
                )
            final_resp["data_correlation_quality"] = json.loads(json_resp)



        res_augi = ez_insight(train_data, outcome, options={})

        insights_df = pd.DataFrame(res_augi['insights']['data'], columns=res_augi['insights']['columns'])
        # insights_df1 = insights_df[insights_df[outcome] == '1']
        # insights_df0 = insights_df[insights_df[outcome] == '0']

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