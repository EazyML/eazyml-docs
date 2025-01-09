"""
This API allows users to build machine learning models.
"""
import os, sys
import json
import traceback
import pandas as pd
from functools import partial

from .globals import (
    global_var as g,
    transparency as tr,
    transparency_api as tr_api
)

from .src.utils import (
            utility,
            api_utils
)
from .src.build_model import (
            helper as build_model_helper
)
from .src.test_model import (
    helper as test_helper
)
from .src.utils.utility_libs import (
                    display_df,
                    display_json,
                    display_md
)

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

convert_json = partial(json.dumps, indent=4, sort_keys=True, default=str)

from .license.license import (
        validate_license,
        init_eazyml
)

def ez_init(license_key: str):
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
    if license_key :
        os.environ["EAZYML_LICENSE_KEY"] = license_key
        # update api and user info in hidden files
        approved, msg = init_eazyml(license_key = os.environ["EAZYML_LICENSE_KEY"])
        return {
                "success": approved,
                "message": msg
            }
    else :
        return {
            "success": False,
            "message": "No license key provided"
        }


@validate_license
def ez_init_model(df, options):
    """
    Initialize and build a predictive model based on the provided dataset and options.

    Parameters :
        - **df** (`DataFrame`): A pandas DataFrame containing the dataset for model initialization.
        - **options** (`dict`): A dictionary of options to configure the model initialization process. Supported keys include:
            - **model_type** (`str`): Type of model to build. Options are "predictive", "timeseries", or "augmented intelligence".
            - **accelerate** (`str`): Whether to accelerate the model-building process. Accepts "yes" or "no".
            - **date_time_column** (`str`): Column name representing date/time values.
            - **remove_dependent** (`str`): Command to remove dependent predictors. Accepts "yes" or "no".
            - **derive_numeric** (`str`): Whether to derive numeric predictors. Accepts "yes" or "no".
            - **derive_text** (`str`): Whether to derive text-based predictors. Accepts "yes" or "no".
            - **phrases** (`dict`): Dictionary to configure text extraction based on predefined phrases.
            - **text_types** (`dict`): Dictionary specifying text types to derive (e.g., "sentiments").
            - **expressions** (`list`): List of expressions for numeric predictor derivation.
            - **outcome** (`str`): Target variable for the model.

    Returns :
        - **Dictionary with Fields**:
            - `success` (`bool`): Indicates if the model has been successful trained.
            - `message` (`str`): Describes the success or failure of the operation.

    Example:
        .. code-block:: python

            ez_init_model(
                df = pd.DataFrame({...}),
                options = {
                        "model_type": "predictive",
                        "accelerate": "yes",
                        "outcome": "target",
                        "remove_dependent": "no",
                        "derive_numeric": "yes",
                        "derive_text": "no",
                        "phrases": {"*": []},
                        "text_types": {"*": ["sentiments"]},
                        "expressions": []
                    }
            )
    """
    try:
        if not isinstance(options, dict):
            return {"success": False, "message": tr_api.VALID_DATATYPE_DICT.replace("this", "options")}, 422
        #Check for valid keys in the options dict
        for key in options:
            if key not in tr_api.EZ_INIT_MODEL_OPTIONS_KEYS_LIST:
                return convert_json({"success": False, "message": tr_api.INVALID_KEY % (key)}), 422
        #Optional parameters
        if "model_type" in options:
            model = options["model_type"]
        else:
            model = "predictive"
        if "accelerate" in options:
            is_accelerated_required = options["accelerate"]
        else:
            is_accelerated_required = "yes"
        if "date_time_column" in options and options["date_time_column"]:
            date_time_column = options["date_time_column"]
        else:
            date_time_column = "null"
        if "remove_dependent" in options:
            remove_dependent_cmd = options["remove_dependent"]
        else:
            remove_dependent_cmd = "no"
        if "derive_numeric" in options:
            derive_numeric_cmd = options["derive_numeric"]
        else:
            derive_numeric_cmd = "no"
        if "derive_text" in options:
            derive_text_cmd = options["derive_text"]
        else:
            derive_text_cmd = "yes"
        if "phrases" in options:
            concepts_dict = options["phrases"]
        else:
            concepts_dict = {"*":[]}
        if "text_types" in options: 
            derive_text_specific_cols_dict = options["text_types"]
        else:
            derive_text_specific_cols_dict = {"*":["sentiments"]}
        if "expressions" in options:
            expressions_list = options["expressions"]
        else:
            expressions_list = []

        #sub_type = dbutils.get_subscription_type(str(uid)).strip().lower()
        original_list = [g.SENTIMENTS, g.GLOVE, g.TOPIC_EXTRACTION, g.CONCEPT_EXTRACTION]
        #is_text_type_prohibited, prohibited_list = utility.check_if_text_types_allowed_for_subscription(sub_type, derive_text_specific_cols_dict)

        #if is_text_type_prohibited:
        #    return convert_json({"success": False, "message": tr_api.TEXT_TYPES_NOT_ALLOWED % (','.join(str(elem).lower() for elem in list(set(original_list) - set(prohibited_list))))}), 422


        #Check the validity of the datatype and the values
#         if (not isinstance(model, str)) or model.lower() not in ["predictive", "timeseries", "augmented intelligence"]:
#             return {"success": False, "message": tr_api.ERROR_MESSAGE_MODEL_TYPE % ("model_type")}, 422
        if (not isinstance(is_accelerated_required, str)) or is_accelerated_required.lower() not in ["yes", "no"]:
            return {"success": False, "message": tr_api.ERROR_MESSAGE_YES_NO_ONLY % ("accelerate")}, 422
        if (not isinstance(remove_dependent_cmd, str)) or remove_dependent_cmd.lower() not in ["yes", "no"]:
            return {"success": False, "message": tr_api.ERROR_MESSAGE_YES_NO_ONLY % ("remove_dependent")}, 422
        if (not isinstance(derive_numeric_cmd, str)) or derive_numeric_cmd.lower() not in ["yes", "no"]:
            return convert_json({"success": False, "message": tr_api.ERROR_MESSAGE_YES_NO_ONLY % ("derive_numeric")}), 422
        if (not isinstance(derive_text_cmd, str)) or derive_text_cmd.lower() not in ["yes", "no"]:
            return {"success": False, "message": tr_api.ERROR_MESSAGE_YES_NO_ONLY % ("derive_text")}, 422
        if (not isinstance(derive_text_specific_cols_dict, dict)):
            return {"success": False, "message": tr_api.INVALID_DATATYPE_PARAMETER % ("text_types") + tr_api.VALID_DATATYPE % ("dict")}, 422
        if (not isinstance(concepts_dict, dict)):
            return {"success": False, "message": tr_api.INVALID_DATATYPE_PARAMETER % ("phrases") + tr_api.VALID_DATATYPE % ("dict")}, 422
        if (not isinstance(expressions_list, list)):
            return {"success": False, "message": tr_api.INVALID_DATATYPE_PARAMETER % ("expressions") + tr_api.VALID_DATATYPE % ("list")}, 422


        #Check if all the dictionaries present are valid ones
        #if (not utility.check_if_valid_dict_for_api(concepts_dict)) or (not utility.check_if_valid_dict_for_api(derive_text_specific_cols_dict)):
        #    return convert_json({"success": False, "message": tr_api.INVALID_DICT}), 422
       
 
        #Check if the user enters a valid dataset id
        #if not dbutils.check_if_dataset_exists_id(did, uid):
        #    return convert_json({"success": False, "message": tr_api.DATASET_NOT_FOUND}), 422
    
        #Check if outcome is set or not
        #if not dbutils.check_if_outcome_set(uid, did):
        #    return convert_json({"success": False, "message": tr_api.OUTCOME_NOT_SET}), 422

        outcome = options["outcome"]
        extra_info = {}
        extra_info["misc_data"] = {}
        extra_info["misc_data_model"] = {}
        extra_info["model_data"] = {}
        #extra_info["model_type"] = "PR"
        extra_info["g_did_mid"] = g
        extra_info["outcome"] = outcome
        #extra_info["misc_data"]["filtered_data"] = file_path
        #extra_info["misc_data"]["statistics"] = stats
        extra_info["misc_data"]["is_imputation_required"] = False
        #extra_info["misc_data"]["model_type"] = "PR"
        #extra_info["misc_data"]["Data Type"] = dtype
        #extra_info["file_path"] = file_path
        
        dtype_df, ps_df = utility.get_smart_datatypes(df, extra_info)
        # imputation
        date_types = dtype_df.loc[dtype_df[g.DATA_TYPE]
                                    == g.DT_DATETIME][g.VARIABLE_NAME].tolist()
        cat_types = dtype_df.loc[dtype_df[g.DATA_TYPE]
                                   == g.DT_CATEGORICAL][g.VARIABLE_NAME].tolist()
        text_types = dtype_df.loc[dtype_df[g.DATA_TYPE]
                                    == g.DT_TEXT][g.VARIABLE_NAME].tolist()
        num_types = dtype_df.loc[dtype_df[g.DATA_TYPE]
                                   == g.DT_NUMERIC][g.VARIABLE_NAME].tolist()

        if outcome in cat_types:
            # set outcome to "NULL" where outcome in empty
            df[outcome] = df[outcome].fillna("NOT DEFINED")

        # Called in inform statistics
        df = utility.convert_data_types(df, cat_types, num_types, date_types)
        num_df = utility.get_statistics(df[num_types], g.DT_NUMERIC)
        cat_df = utility.get_statistics(df[cat_types], g.DT_CATEGORICAL)
        dt_df = utility.get_statistics(df[date_types], g.DT_DATETIME)
        text_df = utility.get_statistics(df[text_types], g.DT_TEXT)
        res_stats = pd.concat([num_df, cat_df, text_df, dt_df], axis=0, ignore_index=True)

        outcome_type = dtype_df.loc[dtype_df[g.VARIABLE_NAME] == outcome][g.DATA_TYPE].tolist()[0]

        if outcome_type ==  "categorical":
            extra_info["misc_data"]["model_type"] = "CL"
            #extra_info["model_type"] = "CL"
        else:
            extra_info["misc_data"]["model_type"] = "PR"
        
        extra_info["model_type"] = "PR"


        pdata_cat_cols_unique_list = dict()
        for col in cat_types:
            pdata_cat_cols_unique_list[col] = df[col].unique().tolist()

        extra_info["misc_data"][g.STAT] = res_stats
        extra_info["misc_data"][g.PRESUF_DF] = ps_df
        extra_info["misc_data"][g.PDATA_CAT_COLS_UNIQUE_LIST] = pdata_cat_cols_unique_list
        extra_info["misc_data"]["Data Type"] = dtype_df
        
        # Check if the dataset has missing values
        if (extra_info is not None) and ("misc_data" in extra_info):
            misc_data = extra_info["misc_data"]
        else:
            return convert_json({"success": False, "message": "MISC DATA NOT AVAILABLE"}), 422
        
        if misc_data[g.IMPUTATION_REQUIRED]:
            if not g.IS_IMPUTATION_DONE in misc_data:
                return {"success": False, "message": tr_api.DATA_HAS_MISSING_VALUES}, 422
        
        #Fetch the model type from the request and return a error message if appropriate model_type is not present.
        if model.lower() == "predictive":
            model_type = g.MODEL_TYPES["PR"]
        # elif model.lower() == "timeseries":
        #     #check if valid date/time column is present
        #     date_type_list = build_model_helper.get_date_time_cols_list(uid, did, extra_info=extra_info)

        #     #Invalid Date/Time Column Provided
        #     if not date_time_column in date_type_list:
        #         return {"success": False, "message": tr_api.DT_COLUMN_INVALID}, 422
        #     model_type = g.MODEL_TYPES["TS"]
        # else:
        #     model_type = g.MODEL_TYPES["DI"]        
    
        #Generate the model id for the model
        #mid = dbutils.update_model_type(uid, did, model_type, source = "API")
        #mid = str(mid)

        ### Update the model name through API
        #dataset_name = dbutils.get_dataset_name(did)
        #model_name = mid + '_' + dataset_name
        #dbutils.update_model_name(uid, mid, model_name)

        ## Cache g_did_mid and , misc_data_model, model_type and model_data in extra_info
        #if (mid is not None) and (mid != ''):
        #extra_info["model_type"] = dbutils.get_model_type(mid)
        #extra_info["model_data"] = dbutils.get_model_data(mid)
        #extra_info["misc_data_model"] = dbutils.get_misc_data_model(uid, mid)
        #if (did is not None) and (did != ''):
        #extra_info["g_did_mid"] = g

        if model_type == "TS":
            if (extra_info is not None) and ("model_data" in extra_info):
                model_data = extra_info["model_data"]
            else:
                return convert_json({"success": False, "message": "MISC DATA NOT AVAILABLE"}), 422
            model_data[g.DATE_TIME_COLUMN] = date_time_column
            model_data[g.CONFIGURE_SEASONALITY] = False
            #dbutils.update_model_data(uid, did, mid, model_data)
            #if (extra_info is not None) and ("model_data" in extra_info):
            #    extra_info["model_data"] = dbutils.get_model_data(mid)
        
        #dbutils.update_misc_data_model(uid, mid, extra_info["misc_data_model"])
        #If the user wants to accelerate through the process then we have to perform all the operations according to 
        #the options provided by the user and then display the final metrics
        if is_accelerated_required.lower() == "yes":
            #For TS models, we directly build models
            if model_type == "TS":
                return {'success': False, 'message': tr_api.MODEL_BUILD_NOT_POSSIBLE}, 422
                # performance_dict = build_model_helper.build_time_series_models(uid, did, mid, display=False, extra_info=extra_info)
                # if performance_dict is None:
                #     return {'success': False, 'message': tr_api.MODEL_BUILD_NOT_POSSIBLE}, 422
                    
                # return {"success": True, "message": tr_api.MODEL_BUILT, "model_performance": utility.decode_json_dict(performance_dict[g.RIGHT][g.TABLE]), "model_id": mid}, 200

            else:
                vif_threshold = 50
                derived_predictors_threshold = 50
                
                #if model_type == "DI":
                #    remove_dependent_cmd = "yes"

                #Remove dependent predictors according to the user"s command
                if remove_dependent_cmd.lower() == "yes":
                    ret_dict = build_model_helper.inform_removal_of_dependent_predictors(df, cmd="1", display=False, extra_info=extra_info)
                else:
                    ret_dict = build_model_helper.inform_removal_of_dependent_predictors(df, cmd="2", display=False, extra_info=extra_info)
                    
                    
                #if (extra_info is not None) and ("model_data" in extra_info):
                model_data = extra_info["model_data"]
                #else:
                #    model_data = dbutils.get_model_data(mid)
                ##check this
                #model_data[g.DATA_FOR_API] = model_data[g.DATA_AFTER_VIF]
                #dbutils.update_model_data(uid, did, mid, model_data)
                #if (extra_info is not None) and ("model_data" in extra_info):
                #    extra_info["model_data"] = dbutils.get_model_data(mid)

                #Saving the user"s options for numeric derived predictors if numeric columns are present
                if build_model_helper.datatype_col_present(df, extra_info=extra_info):
                    if derive_numeric_cmd.lower() == "yes":
                        ret_dict = build_model_helper.ask_for_derived_predictors(cmd="1", display=False, extra_info=extra_info)
                        is_derived_numeric_possible, derived_df = build_model_helper.derive_numeric_for_api( expressions_list, extra_info=extra_info)
                        if not is_derived_numeric_possible:
                            ret_dict = build_model_helper.ask_for_derived_predictors(df, cmd="2", display=False, extra_info=extra_info)
                    else:
                        ret_dict = build_model_helper.ask_for_derived_predictors(df, cmd="2", display=False, extra_info=extra_info)
                        #return ret_dict, extra_info
                else:
                    ret_dict = build_model_helper.ask_for_derived_predictors(df, cmd="2", display=False, extra_info=extra_info)

                #Saving the user"s options for text derived predictors if text columns are present
                if build_model_helper.datatype_col_present(df, g.DT_TEXT, extra_info=extra_info):
                    if derive_text_cmd.lower() == "yes":
                        ret_dict = build_model_helper.ask_for_derived_text_predictors(cmd="1", display=False, extra_info=extra_info)
                        is_derived_text_possible, derived_df = build_model_helper.derive_text_for_api(concepts_dict, derive_text_specific_cols_dict, extra_info=extra_info)
                        if not is_derived_text_possible:
                            ret_dict = build_model_helper.ask_for_derived_text_predictors(cmd="2", display=False, extra_info=extra_info)
                    else:
                        ret_dict = build_model_helper.ask_for_derived_text_predictors( cmd="2", display=False, extra_info=extra_info)
                else:
                    ret_dict = build_model_helper.ask_for_derived_text_predictors(cmd='2', display=False, extra_info=extra_info)
                
                #Feature extraction
                is_feature_selection_possible, selected_features_list, selected_score_list, extra_info = build_model_helper.feature_extraction_for_api(df, extra_info=extra_info)

                if not is_feature_selection_possible:
                    return {'success': False, 'message': 'Feature selection is not possible as there is no numeric columns left after encoding.'}, 422
               
            
                #return extra_info, status_code
                #Build Models
                var_type = misc_data[g.DATA_TYPE]
                cat_types = var_type.loc[var_type[g.DATA_TYPE] == g.DT_CATEGORICAL][g.VARIABLE_NAME].tolist()
                if extra_info["outcome"] in cat_types:
                    cat_types.remove(extra_info["outcome"])
                df = utility.create_dummy_features(df, cat_types)
                if not g.API_FLEXIBILITY:
                    performance_dict = build_model_helper.build_model_show_core_predictors(df, display=False, extra_info=extra_info)
                    try:
                        performance_dict = json.loads(performance_dict.get_data())
                    except Exception as e:
                        pass
                    if 'invalid_state' in performance_dict:
                        return {"success": False, "message": performance_dict['left']['body']}, 422

                else:
                    return {'success': False, 'message': tr_api.MODEL_BUILD_NOT_POSSIBLE}, 422
                    # is_model_build_possible, performance_dict, message = build_model_helper.build_model_for_api(uid, did, mid, [], extra_info=extra_info)
                    # try:
                    #     performance_dict = json.loads(performance_dict.get_data())
                    # except Exception as e:
                    #     pass
                    # if 'invalid_state' in performance_dict:
                    #     return {"success": False, "message": performance_dict['left']['body']}, 422
                    # if not is_model_build_possible:
                    #     return {'success': False, 'message': message}, 422
                
                global_importance_df = build_model_helper.show_core_predictors(cmd="", display=True, return_df=True, extra_info=extra_info)
                global_importance_dict_to_be_returned = dict()
                global_importance_dict_to_be_returned["data"] = global_importance_df.values.tolist()
                global_importance_dict_to_be_returned["columns"] = global_importance_df.columns.tolist()
                #Return Model scores and global importance values
                output_data = api_utils.output_extra_info(extra_info)
                output_data = api_utils.encrypt_dict(output_data)
                return {"success": True, "message": tr_api.MODEL_BUILT, "model_performance": utility.decode_json_dict(performance_dict[g.RIGHT][g.TABLE]), "global_importance": global_importance_dict_to_be_returned, "extra_info": output_data}, 200

        # else:
        #     #Initialize DATA_FOR_API key in model_data. This key will be used henceforth in fetching the dataframe for the different oper            #ations like remove_dependent, derive_features, feature_selection, model_building. This is done to overcome the issue of
        #     #users randomly calling the api"s in no specific order.
        #     if model_type == "PR" or model_type == "DI":
        #         pdata_original = dbutils.get_processed_data(uid, did)
        #         if (extra_info is not None) and ("misc_data" in extra_info):
        #             misc_data = extra_info["misc_data"]
        #         else:
        #             misc_data = dbutils.get_misc_data(uid, did)
        #         if (extra_info is not None) and ("model_data" in extra_info):
        #             model_data = extra_info["model_data"]
        #         else:
        #             model_data = dbutils.get_model_data(mid)
        #         outcome = dbutils.get_outcome_data(uid, did)
        #         data_for_api, var_type, added_columns = build_utils.get_processed_data_for_api(misc_data, pdata_original, outcome)
        #         model_data[g.DATA_FOR_API] = data_for_api
        #         model_data[g.UPDATE_DT_TYPES] = var_type
        #         #dbutils.update_model_data(uid, did, mid, model_data)
        #         #if (extra_info is not None) and ("model_data" in extra_info):
        #         #    extra_info["model_data"] = dbutils.get_model_data(mid)
        #     return {"success": True, "message": tr_api.MODEL_BUILDING_REGISTERED, "model_id": mid}, 200
    except Exception as e:
        # print(traceback.print_exc())
        #api_logging.dbglog(("Exception in ez_model", e))
        return convert_json({"success": False, "message": tr_api.INTERNAL_SERVER_ERROR}), 500


@validate_license
def ez_predict(test_data, options):
    """
    Perform prediction on the given test data based
    on model options and validate the input dataset.

    Parameters :
        - **test_data** (`DataFrame`): The test dataset to be evaluated. It must have consistent features with the trained model.
        - **options** (`dict`): A dictionary of options to configure the model initialization process. Supported keys include:
            - **extra_info** (`dict`): Contains encrypted or unencrypted details about the model and environment.
            - **model** (`str`): Specifies the model to be used for prediction. If not provided, the default model from `extra_info` is used.
            - **outcome** (`str`): Target variable for the model.

    Returns :
        - **tuple**:
            A tuple consisting of:
                - dict or pandas.DataFrame : If successful, returns the prediction results in a DataFrame. 
                - In case of failure, returns a dictionary with the keys:
                    - "success" (`bool`) : Indicates if the operation was successful.
                    - "message" (`str`) : Contains an error or informational message.
                - int : HTTP status code (200 for success, 422 for errors).
    """
#     try:
    global g
    extra_info = options["extra_info"]
    outcome = options["outcome"]
        
    try:
        extra_info = api_utils.decrypt_dict(extra_info)
    except:
        pass
    extra_info["g_did_mid"] = g
    
    model_data = extra_info["model_data"]
    models_list = model_data[g.METRICS]["Model"].values.tolist()

    if "model" in options:
        model = options["model"]
    else:
        model = extra_info["model_data"]["metrics"]["Model"][0]
        
    extra_info["model_data"]["predict_model"] = model
    difference = test_helper.check_if_test_data_is_consistent(test_data, extra_info=extra_info)
    only_extra = False
    missing = False
    if difference is not None:
        if 'missing' in difference:
            message = tr.ABSENT_COL_HEAD + '\n'
            message += ', '.join(difference['missing'])
            message += '\n'
            missing = True

        if 'extra' in difference:
            message = tr.EXTRA_COL_HEAD + '\n'
            message += ', '.join(difference['extra'])
            message += '\n'
            if not missing:
                only_extra = True
        if not only_extra:
            message += tr.REUPLOAD_DATA
            #build_model_helper.update_model_response_cache(uid, did, mid, "null", "null", "null", extra_info=extra_info)
            return {"success": False,"message": tr_api.INCONSISTENT_DATASET}, 422

    test_data, col_name = test_helper.process_test_data(test_data, extra_info=extra_info)
    if test_data is None:
        return {"success": False,"message": tr.TRAIN_TEST_COLUMN_MISMATCH % col_name},422
    elif test_data.empty:
        return {"success": False, "message": tr.INVALID_VALUE}, 422
    res = test_helper.show_results(test_data, display=True, extra_info=extra_info)
    if 'invalid_state' in res:
        return {"success": False, "message": res['left']['body']}, 422
    if isinstance(res, pd.DataFrame):
        return res, 200
    else:
        return {"success": False, "message": res['left']['body']}, 422


@validate_license
def ez_display_json(resp):
    """
    Function to display formatted json
    """
    return display_json(resp)


@validate_license
def ez_display_df(resp):
    """
    Function to display formatted dataframe
    """
    return display_df(resp)


@validate_license
def ez_display_md(resp):
    """
    Function to display formatted markdown
    """
    return display_md(resp)
