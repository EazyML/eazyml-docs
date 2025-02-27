"""
EazyML Augmented Intelligence extract insights from Dataset with certain insights
score which is calculated using coverage of that insights.
"""
import os
import pandas as pd
from .globals import (
    transparency_api as tr_api,
    vars as g
)

from .src import utils
from .license import (
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


def ez_insight(train_data, outcome,
            options={}):
    """
    Fetch insights from the input training data based on the mode, outcome, and options. 
    Supports classification and regression tasks.

    Parameters :
        - **train_data** (str/DataFrame):
            Path to the training data file.
        - **outcome** (str):
            The target variable in the training data.
        - **options** (dict, optional):
            Additional options for augmented intelligence. Default is an empty dictionary. Supported keys include:
                - "data_source" (str): Specifies the data source type (e.g., "parquet" or "system").

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
                "message": "Insights fetched successfully",
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

            ez_insight(
                train_data='train.csv',
                outcome='target'
            )
    """
    try:
        data_source = "system"


        if isinstance(train_data, str):
            if not os.path.exists(train_data):
                return {
                    "success": False,
                    "message": "train_file_path does not exist."
                }
            train_data, _ = utils.get_df(train_data, data_source=data_source)
        elif isinstance(train_data, pd.DataFrame):
            train_data = train_data
        else :
            return {
                "success": False,
                "message": 'train_data should be of either string or DataFrame'
                }
        #Check for valid keys in the options dict
        is_list = lambda x: type(x) == list
        is_string = lambda x: isinstance(x, str)
        if not is_string(outcome):
            return {
                        "success": False,
                        "message": tr_api.ALL_STR_PARAM
                    }
        if outcome not in train_data.columns:
            return {
                "success": False,
                "message": "Outcome is not present in training data columns"}

        mode, data_type_dict, selected_features_list =\
            utils.get_mode_data_type_selected_features(train_data, outcome)

        if not isinstance(data_type_dict, dict):
            return {
                    "success": False,
                    "message": tr_api.VALID_DATATYPE_DICT.replace(
                        "this", "options"),
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
        if ("data_source" in options and options[
            "data_source"] == "parquet"):
            data_source = "parquet"

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
        if mode not in ['classification', 'regression']:
            return {
                "success": False,
                "message": "Please provide valid mode.('classification'/'regression')"
                   }
        if mode == 'regression' and train_data[
            outcome].dtype == 'object':
            return {
                    "success": False,
                    "message": "The type of the outcome column is a string, so the mode should be classification."
                    }
        if mode == 'classification' and pd.api.types.is_float_dtype(
            train_data[outcome]):
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
        if not is_string(mode):
            return {
                        "success": False,
                        "message": tr_api.ALL_STR_PARAM
                    }

        for key in options:
            if key not in tr_api.EZ_BUILD_MODELS_OPTIONS_KEYS_LIST:
                return {
                    "success": False, "message": tr_api.INVALID_KEY % (key)}

        train_data, global_info_dict, cat_list =\
            utils.preprocessing_steps(
            train_data, data_type_dict, outcome)
        for col in selected_features_list:
            if col not in train_data.columns.tolist():
                return {
                        "success": False,
                        "message": "Please provide valid column name in selected_features_list"
                        }
        train_data, rule_lime_dict = utils.processing_steps(
            train_data, global_info_dict, selected_features_list)

        user_features_list = train_data.columns.tolist()

        if (not isinstance(user_features_list, list)):
            return {"success": False, "message": tr_api.INVALID_DATATYPE_PARAMETER % ("features") + tr_api.VALID_DATATYPE % ("list")}

        ## Cache g, g_did_mid, misc_data, misc_data_model,
        ## model_data and model_type in extra_info
        extra_info = dict()
        extra_info["g"] = g
        extra_info["var_type"] = data_type_dict
                
        is_model_build_possible, performance_dict, message =\
            utils.build_model_for_api(train_data, mode, outcome,
            	user_features_list, extra_info=extra_info)
        if not is_model_build_possible:
            return {'success': False, 'message': message}
        performance_dict.drop(['Thresholds'], axis='columns', inplace=True)
        performance_dict_to_be_returned = dict()
        performance_dict_to_be_returned[
            "data"] = performance_dict.values.tolist()
        performance_dict_to_be_returned[
            "columns"] = performance_dict.columns.tolist()
        return {
                "success": True,
                "message": 'Insights have been fetched successfully',
                "insights": performance_dict_to_be_returned
               }
    except Exception as e:
        print(e)
        return {"success": False, "message": tr_api.INTERNAL_SERVER_ERROR}


def ez_validate(train_data, outcome, insights, test_data,
                options={}):
    """
    Validate Augmented Intelligence insights on test data, based on mode, outcome, and options.
    Supports classification and regression tasks.

    Parameters :
        - **mode** (str):
            The type of problem. Must be either 'classification' or 'regression'.
        - **outcome** (str):
            The target variable in both training and test data.
        - **insights** (dict):
            Augmented Intelligence insights provided by ez_insight.
        - **train_file_path** (str):
            Path to the train data file.
        - **test_file_path** (str):
            Path to the test data file.
        - **data_type_dict** (dict):
            Dictionary which contain type of each feature.
        - **options** (dict, optional):
            Additional options for augmented intelligence. Default is an empty dictionary. Supported keys include:
                - "data_source" (str): Specifies the data source type (e.g., "parquet" or "system").
                - "record_number" (list): The record from the insight list whose validation needs to be explained.

    Returns :
        - **Dictionary with Fields** :
            - **success** (bool): Indicates whether the operation was successful.
            - **message** (str): Describes the outcome or error message.
            - **validations** (dict, optional): A Pandas Dataframe in JSON format. The JSON contains two keys:
                - "data" (str): The data in list of list format.
                - "columns" (str): The columns of the dataframe in list format.
            - **validation_filter** (dict, optional): Filtered test data for given record numbers.

        **On Success** :  
        A JSON response with
        
        .. code-block:: json

            {
                "success": true,
                "message": "Insights validated successfully",
                "validations": {
                    "data": [".."],
                    "columns": [".."]
                },
                "validation_filter": {
                    "record_number": "..",
                    "filtered_data": {".."},
                    "Augmented Intelligence Insights": "..",
                    "Insight Scores": ".."
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
        - Verifies that `outcome` exists as a column in both training and test data.
        - Checks that `options` is a dictionary and contains valid keys.
        - Validates data types for `mode`, `outcome`, `train_file_path` and `test_file_path` (must all be strings).

    Steps :
        1. Load both training and test data based on the specified `data_source`.
        2. Validates input parameters for correctness.
        3. Extracts user-specified features or defaults to all features in the data.
        4. Calls `build_model_for_api` to build the model and obtain its performance metrics.
        5. Processes performance metrics into a returnable dictionary format.

    Notes :
        - If model building fails, returns a failure message with the reason.
    
    Example:
        .. code-block:: python

            ez_validate(
                mode='classification',
                outcome='target',
                insights=insights,
                train_file_path='train.csv',
                test_file_path='test.csv',
                data_type_dict=data_type_dict,
                options={"data_source": "parquet", "record_number": [1, 2, 3]}
            )
    """
    try:
        data_source = "system"
        if isinstance(train_data, str):
            if not os.path.exists(train_data):
                return {
                    "success": False,
                    "message": "train_file_path does not exist."
                }
            train_data, _ = utils.get_df(train_data, data_source=data_source)
        elif isinstance(train_data, pd.DataFrame):
            train_data = train_data
        else:
            return {
                "success": False,
                "message": 'train_data should be of either string or DataFrame'
            }
        if isinstance(test_data, str):
            if not os.path.exists(test_data):
                return {
                    "success": False,
                    "message": "test_file_path does not exist."
                }
            test_data, _ = utils.get_df(test_data, data_source=data_source)
        elif isinstance(test_data, pd.DataFrame):
            test_data = test_data
        else:
            return {
                "success": False,
                "message": 'test_data should be of either string or DataFrame'
            }
        is_list = lambda x: type(x) == list
        is_string = lambda x: isinstance(x, str)
        if not is_string(outcome):
            return {
                        "success": False,
                        "message": tr_api.ALL_STR_PARAM
                    }
        if outcome not in train_data.columns or\
            outcome not in test_data.columns:
            return {
                "success": False,
                "message": "Outcome is not present in either training or test data columns"}

        mode, data_type_dict, _ =\
            utils.get_mode_data_type_selected_features(train_data, outcome)

        if not isinstance(insights, dict):
            return {
                    "success": False,
                    "message": "Please provide insights which you got from ez_insight"
                    }
        if not isinstance(data_type_dict, dict):
            return {
                    "success": False,
                    "message": tr_api.VALID_DATATYPE_DICT.replace(
                        "this", "options"),
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
        for col in set(insights.keys()):
            if col not in ['data', 'columns']:
                return {
                        "success": False,
                        "message": "Please provide insights which you got from ez_insight"
                        }
        insight_columns = [outcome, 'Augmented Intelligence Insights',
            'Insight Scores']
        if insights['columns'] != insight_columns:
            return {
                    "success": False,
                    "message": "Please provide insights which you got from ez_insight"
                    }
        if ("data_source" in options and options[
            "data_source"] == "parquet"):
            data_source = "parquet"


        for col in data_type_dict.keys():
            if col not in train_data.columns or col not in test_data.columns:
                return {
                        "success": False,
                        "message": col + " is not present in either training or test data columns"
                        }
        if len(data_type_dict.keys()) < 2:
            return {
                    "success": False,
                    "message": "Please provide data type for all columns (on which insights are fetched) in data_type."
                    }
        if mode not in ['classification', 'regression']:
            return {
                "success": False,
                "message": "Please provide valid mode.('classification'/'regression')"
                   }
        if mode == 'regression' and (
            test_data[outcome].dtype == 'object' or train_data[
            outcome].dtype == 'object'):
            return {
                    "success": False,
                    "message": "The type of the outcome column is a string, so the mode should be classification."
                    }
        if mode == 'classification' and (pd.api.types.is_float_dtype(
            test_data[outcome]) or pd.api.types.is_float_dtype(
            train_data[outcome])):
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
        if not is_string(mode):
            return {
                        "success": False,
                        "message": tr_api.ALL_STR_PARAM
                    }
        for key in options:
            if key not in tr_api.EZ_VALIDATE_OPTIONS_KEYS_LIST:
                return {
                    "success": False, "message": tr_api.INVALID_KEY % (key)}
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
                rec_n = utils.get_records_list(record_number)
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
            insights_count = len(insights['data'])
            for rec_number in record_number:
                if int(rec_number) > insights_count:
                    return {
                            "success": False,
                            "message": "'record_number' in the 'options' parameter has values more than number of rows in the augi_insights."
                            }
        else:
            record_number = [1]
        if (len(test_data) < g.BASIC_TEST_DATASET_COUNT):
            return {"success": False, "message": tr_api.TEST_DATASET_SIZE_LOW}
        if mode == 'classification':
            unique_values = train_data[outcome].unique()
            unique_values = [str(i) for i in unique_values]
        for sublist in insights['data']:
            if len(sublist) != 3:
                return {
                        "success": False,
                        "message": "Please provide insights which you got from ez_insight"
                        }
            if mode == 'classification' and sublist[0] not in unique_values:
                return {
                        "success": False,
                        "message": "Please provide insights which you got from ez_insight"
                        }
            if pd.api.types.is_float_dtype(sublist[2]):
                return {
                        "success": False,
                        "message": "Please provide insights which you got from ez_insight"
                        }
        extra_info = dict()
        extra_info["g"] = g
        name_dt_df = pd.DataFrame()
        name_dt_df[g.VARIABLE_NAME] = data_type_dict.keys()
        name_dt_df[g.DATA_TYPE] = data_type_dict.values()
        extra_info[g.DATA_TYPE] = name_dt_df
        difference = utils.check_if_test_data_is_consistent(
            train_data, test_data, outcome, extra_info=extra_info
        )
        only_extra = False
        missing = False
        message = ''
        if difference is not None:
            if 'missing' in difference:
                message += tr_api.ABSENT_COL_HEAD + '\n'
                message += ', '.join(difference['missing'])
                message += '\n'
                missing = True

            if 'extra' in difference:
                message += tr_api.EXTRA_COL_HEAD + '\n'
                message += ', '.join(difference['extra'])
                message += '\n'
                if not missing:
                    only_extra = True
            if not only_extra:
                message += tr_api.REUPLOAD_DATA
                return {
                        "success": False,
                        "message": tr_api.INCONSISTENT_DATASET,
                       }
        if not g.AUGI_VALIDATE_TRAIN:
            test_data, col_name = utils.process_test_data(
                test_data, extra_info=extra_info)
            if test_data is None:
                return {
                        "success": False,
                        "message": tr_api.TRAIN_TEST_COLUMN_MISMATCH % col_name
                       }
            elif test_data.empty:
                return {
                        "success": False,
                        "message": tr_api.INVALID_VALUE
                       }

            #Store the metadata and the test data
            model_data = {}
            model_data[g.IS_TEST_TYPE_ONLINE] = False
            model_data[g.TEST_DATA] = test_data
            extra_info["model_data"] = model_data

            # Dropping missing values
            num_types = name_dt_df.loc[name_dt_df[
                g.DATA_TYPE] == g.DT_NUMERIC][g.VARIABLE_NAME].tolist()
            cat_types = name_dt_df.loc[name_dt_df[
                g.DATA_TYPE] == g.DT_CATEGORICAL][g.VARIABLE_NAME].tolist()
            date_types = name_dt_df.loc[name_dt_df[
                g.DATA_TYPE] == g.DT_DATETIME][g.VARIABLE_NAME].tolist()
            text_types = name_dt_df.loc[name_dt_df[
                g.DATA_TYPE] == g.DT_TEXT][g.VARIABLE_NAME].tolist()
            pdata_cat_cols_unique_list = dict()
            # Just drop missing values instead of imputing them
            test_data = test_data.dropna(
                subset=num_types+date_types).reset_index(drop=True)
            if not (outcome in num_types and outcome in date_types):
                test_data = test_data.dropna(subset=[
                    outcome]).reset_index(drop=True)
            if test_data is None or test_data.shape[0] <= 0:
                return {
                        "success": False,
                        "message": tr_api.EMPTY_TEST_DATASET
                       }
            if (len(test_data) < g.BASIC_TEST_DATASET_COUNT):
                return {
                        "success": False,
                        "message": tr_api.MINIMUM_ROWS_NOT_PRESENT
                       }

            test_data = utils.fetch_test_data_for_removing_outliers(
                train_data, test_data, outcome, extra_info=extra_info)
            #calculating validation scores
            res = utils.calculate_validation_score(
                mode, test_data, outcome, insights,
                extra_info=extra_info)
        else:
            test_data = train_data
            model_data = {}
            test_data, _  = utils.process_test_data(
                test_data, extra_info=extra_info)
            model_data[g.IS_DUPLICATE] = True
            extra_info["model_data"] = model_data
            test_data = utils.fetch_test_data_for_removing_outliers(
                train_data, test_data, outcome, extra_info=extra_info)
            #calculating validation scores
            res = utils.calculate_validation_score(
                mode, test_data, outcome, insights,
                extra_info=extra_info)
        if 'invalid_state' in res:
            return {
                    "success": False,
                    "message": res['left']['body']
                   }
        # To fetch the validation
        if g.RIGHT in res and g.TABLE in res[g.RIGHT]:
            message = tr_api.VALIDATION_SUCCESS
            results = []
            for rec_number in record_number:
                api_dict = dict()
                api_dict["record_number"] = rec_number
                api_dict["Insight Scores"] = res[g.RIGHT][g.TABLE][
                    'data'][rec_number-1][3]
                if g.AUGI_VALIDATION_SCORES:
                    api_dict["Validation Scores"] = res[g.RIGHT][g.TABLE][
                        'data'][rec_number-1][4]
                api_dict["Augmented Intelligence Insights"] = res[g.RIGHT][
                    g.TABLE]['data'][rec_number-1][2]
                api_dict["filtered_data"] = utils.get_json_dict(res[g.RIGHT][
                    'Filtered_Data'][rec_number-1])
                results.append(api_dict)
            return {
                    "success": True,
                    "message": tr_api.DATASET_UPLOAD_SUCCESS + ' ' + message,
                    "validations": res[g.RIGHT][g.TABLE],
                    "validation_filter": results
                   }
        return {
                "success": False,
                "message": tr_api.VALIDATION_FAILED
               }
    except Exception as e:
        return {
                "success": False,
                "message": tr_api.INTERNAL_SERVER_ERROR
               }
