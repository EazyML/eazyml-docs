"""
EazyML Augmented Intelligence extract insights from Dataset with certain insights
score which is calculated using coverage of that insights.
"""
from .transparency_api import (
    INTERNAL_SERVER_ERROR, INVALID_DATATYPE_PARAMETER,
    EZ_BUILD_MODELS_OPTIONS_KEYS_LIST, VALID_DATATYPE,
    VALID_DATATYPE_DICT, ALL_STR_PARAM, INVALID_KEY
)

from .utils import get_df, build_model_for_api

def ez_augi(mode, outcome, train_file_path, options={}):
    """
    Builds a predictive model based on the input training data, mode, and options. 
    Supports classification and regression tasks.

    Parameters :
        - **mode** (str):
            The type of model to build. Must be either 'classification' or 'regression'.
        - **outcome** (str):
            The target variable in the training data.
        - **train_file_path** (str):
            Path to the training data file.
        - **options** (dict, optional):
            Additional options for model building. Default is an empty dictionary. Supported keys include:
                - "data_source" (str): Specifies the data source type (e.g., "parquet" or "system").
                - "features" (list): A list of feature column names to use for model building.

    Returns :
        - **Dictionary with Fields** :
            - **success** (bool): Indicates whether the operation was successful.
            - **message** (str): Describes the outcome or error message.
            - **insights** (dict, optional): Contains model performance data such as insights and insight-score if the operation was successful.

        **On Success** :  
        A JSON response with
        
        .. code-block:: json

            {
                "success": true,
                "message": "Explanation generated successfully",
                "insights": {
                    "data": [".."],
                    "columns": [".."]
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
            

    Validation :
        - Ensures the `mode` is either 'classification' or 'regression'.
        - Verifies that `outcome` exists as a column in the training data.
        - Checks that `options` is a dictionary and contains valid keys.
        - Validates data types for `mode`, `outcome`, and `train_file_path` (must all be strings).
        - Ensures "features" in `options`, if provided, is a list.

    Steps :
        1. Loads the training data based on the specified `data_source`.
        2. Validates input parameters for correctness.
        3. Extracts user-specified features or defaults to all features in the data.
        4. Calls `build_model_for_api` to build the model and obtain its performance metrics.
        5. Processes performance metrics into a returnable dictionary format.

    Notes :
        - If model building fails, returns a failure message with the reason.
        - Drops "Thresholds" column from the performance metrics before returning insights.
    
    Example:
        .. code-block:: python

            ez_augi(
                mode='classification',
                outcome='target',
                train_file_path='train.csv',
            )
    """
    try:
        data_source = "system"
        if ("data_source" in options and options[
            "data_source"] == "parquet"):
            data_source = "parquet"
        train_data, _ = get_df(train_file_path, data_source=data_source)
        if outcome not in train_data.columns:
            return {
                "success": False,
                "message": "Outcome is not present in training data columns"}
        if mode not in ['classification', 'regression']:
            return {
                "success": False,
                "message": "Please provide valid mode.('classification'/'regression')"}
        if not isinstance(options, dict):
            return {
                    "success": False,
                    "message": VALID_DATATYPE_DICT.replace(
                        "this", "options"),
                    }
        #Check for valid keys in the options dict
        is_list = lambda x: type(x) == list
        is_string = lambda x: isinstance(x, str)
        if (
            not is_string(mode)
            or not is_string(outcome)
            or not is_string(train_file_path)
        ):
            return {
                        "success": False,
                        "message": ALL_STR_PARAM
                    }

        for key in options:
            if key not in EZ_BUILD_MODELS_OPTIONS_KEYS_LIST:
                return {
                    "success": False, "message": INVALID_KEY % (key)}

        if "features" in options:
            user_features_list = options["features"]
        else:
            user_features_list = train_data.columns.tolist()

        if (not isinstance(user_features_list, list)):
            return {"success": False, "message": INVALID_DATATYPE_PARAMETER % ("features") + VALID_DATATYPE % ("list")}

        ## Cache g, g_did_mid, misc_data, misc_data_model, model_data and model_type in extra_info
        extra_info = dict()
        extra_info["g"] = g
                
        is_model_build_possible, performance_dict, message =\
            build_model_for_api(train_data, mode, outcome,
            	user_features_list, extra_info=extra_info)
        if not is_model_build_possible:
            return {'success': False, 'message': message}
        # global_importance_df = show_core_predictors(cmd="", display=True, return_df=True, extra_info=extra_info)
        # global_importance_dict_to_be_returned = dict()
        # global_importance_dict_to_be_returned["data"] = [] #global_importance_df.values.tolist()
        # global_importance_dict_to_be_returned["columns"] = [] #global_importance_df.columns.tolist()
        performance_dict.drop(['Thresholds'], axis='columns', inplace=True)
        performance_dict_to_be_returned = dict()
        performance_dict_to_be_returned["data"] = performance_dict.values.tolist()
        performance_dict_to_be_returned["columns"] = performance_dict.columns.tolist()
        return {"success": True, "message": 'Insights have been fetched successfully', "insights": performance_dict_to_be_returned}
        #         "global_importance": global_importance_dict_to_be_returned}
    except Exception as e:
        print(e)
        return {"success": False, "message": INTERNAL_SERVER_ERROR}

