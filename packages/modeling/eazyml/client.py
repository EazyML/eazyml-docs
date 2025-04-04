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
    transparency_api as tr_api,
    config
)

from .src.utils import (
            utility,
            api_utils, 
            spark_utils,
            upload_utils
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

from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from .globals import logger as log
log.initlog()
# make level for pyj4 log as critical
log_inst = log.instance()
log_inst.getLogger("pyj4").setLevel(log_inst.CRITICAL)

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

            from eazyml import ez_init

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


def ez_build_model(train_data, outcome, options={}):
    """
    Initialize and build a predictive model based on the provided dataset and options.

    Args:
        - **train_data** (`DataFrame` or `str`): A pandas DataFrame containing the dataset for model initialization. Alternatively, you can provide the file path of the dataset (as a string).
        - **outcome** (`str`): The target variable for the model.
        - **options** (`dict`, optional): A dictionary of options to configure the model initialization process. If not provided, the function will use default settings. Supported keys include:
            
            - **model_type** (`str`, optional): Specifies the type of model to build. The supported value is "predictive".
            - **spark_session** (`SparkSession` or `None`, optional): If a Spark session is provided, distributed computation will be used. If `None`, standard computation is used.

    Returns:
        A dictionary containing the results of the model building process with the following fields:
            
            - **success** (`bool`): Indicates whether the model was successfully trained.
            - **message** (`str`): A message describing the success or failure of the operation.

            **On Success**:
            
            - **model_performance** (`DataFrame`): A DataFrame providing the performance metrics of the trained model(s).
            - **global_importance** (`DataFrame`): A DataFrame containing the feature importance scores.
            - **model_info** (`Bytes`): Encrypted model information that will be used by `ez_predict` for making predictions on test data.

    Note:
        - Please save the `response` obtained after building the model and provide the `model_info` to the `ez_predict` function for making predictions on test data.
        - If you are using a `spark_session`, save the necessary Spark models separately from the `model_info` and pass them as `spark_model` in the `options` dictionary when calling `ez_predict`, along with the session and `model_info`.
        - Since Spark models cannot be directly saved in the `model_info` output, you must manually save the individual models from `response["model_info"]["spark_module"]["Models"][index]["model"]` for each index. Use the `Pipeline` module to save and load the models as needed.

    Example:
        .. code-block:: python

            import pandas as pd
            import joblib
            from eazyml import ez_build_model

            # Load the training data (make sure the file path is correct).
            train_file_path = "path_to_your_training_data.csv"  # Replace with the correct file path
            train_data = pd.read_csv(train_file_path)

            # Define the outcome (target variable) for the model
            outcome = "target"  # Replace with your actual target variable name

            # Set the options for building the model
            build_options = {"model_type": "predictive"}

            # Call the eazyml function to build the model
            build_response = ez_build_model(train_data, outcome, options=build_options)

            # build_response is a dictionary object with following keys.
            # print(build_response.keys())
            # dict_keys(['success', 'message', 'model_performance', 'global_importance', 'model_info'])

            # Save the response for later use (e.g., for predictions with ez_predict)
            build_model_response_path = 'model_response.joblib'
            joblib.dump(build_response, build_model_response_path)

    """
    
    try:
        log.log_db("Initialze ez_build_model")
        #Check for valid keys in the options dict
        for key in options:
            if key not in tr_api.EZ_INIT_MODEL_OPTIONS_KEYS_LIST:
                return {"success": False, "message": tr_api.INVALID_KEY % (key)}
            
        if isinstance(train_data, str):
            if not os.path.exists(train_data):
                return {"success": False,
                        "message": "train_data provided path does not exist."}
            else:
                train_file_path = train_data
                if "spark_session" in options:
                    spark = options["spark_session"]
                else:
                    spark = None
                if spark:
                    try:
                        spark_version = spark.version
                        return {"success": False, "message": "This version currently does not support the spark module for building models."}
                    except:
                        return {"success": False, "message": tr_api.SPARK_SESSION}
                    train_data = spark_utils.get_df_spark(train_file_path, spark)
                else:
                    train_data = upload_utils.get_df(train_file_path)

            if train_data is None:
                return {"success": False, "message": tr_api.VALID_DATAFILEPATH.replace("this", "train_data")}         
        elif not isinstance(train_data, (pd.DataFrame, DataFrame)):
            return {"success": False, "message": tr_api.VALID_DATAOBJECT.replace("this", "train_data")}
        elif isinstance(train_data, DataFrame):
            if "spark_session" in options:
                spark = options["spark_session"]
            else:
                spark = None
            if spark:
                try:
                    spark_version = spark.version
                except:
                    return {"success": False, "message": tr_api.SPARK_SESSION}
                return {"success": False, "message": "This version currently does not support the spark module for building models."}
        if not isinstance(options, dict):
            return {"success": False, "message": tr_api.VALID_DATATYPE_DICT.replace("this", "options")}
        
        
            
        upgrade_required = ["remove_dependent", "derive_numeric", "derive_text", "phrases", "text_types", "expressions"]
        
        # Get the keys that are present in both 'options' and 'upgrade_required'
        present_keys = [key for key in options if key in upgrade_required]

        # Check and print the result
        for each_key in present_keys:
            if options[each_key] == "yes":
                return {"success": False, "message": tr_api.OPTIONS_LIMITED.replace("this", each_key)}
        
        #Optional parameters
        if "model_type" in options:
            model = options["model_type"]
        else:
            model = "predictive"
            
        if model != "predictive":
            return {"success": False, "message": tr_api.MODEL_TYPE.replace("this", "model_type")}
        
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
        
        if "spark_session" in options:
            spark = options["spark_session"]
            log.log_db(f"Spark Session has been provided {spark}")
        else: 
            spark = None
            log.log_db(f"Standard Modelling process")
        # original_list = [g.SENTIMENTS, g.GLOVE, g.TOPIC_EXTRACTION, g.CONCEPT_EXTRACTION]
        
        if (not isinstance(is_accelerated_required, str)) or is_accelerated_required.lower() not in ["yes", "no"]:
            return {"success": False, "message": tr_api.ERROR_MESSAGE_YES_NO_ONLY % ("accelerate")}
        if (not isinstance(remove_dependent_cmd, str)) or remove_dependent_cmd.lower() not in ["yes", "no"]:
            return {"success": False, "message": tr_api.ERROR_MESSAGE_YES_NO_ONLY % ("remove_dependent")}
        if (not isinstance(derive_numeric_cmd, str)) or derive_numeric_cmd.lower() not in ["yes", "no"]:
            return {"success": False, "message": tr_api.ERROR_MESSAGE_YES_NO_ONLY % ("derive_numeric")}
        if (not isinstance(derive_text_cmd, str)) or derive_text_cmd.lower() not in ["yes", "no"]:
            return {"success": False, "message": tr_api.ERROR_MESSAGE_YES_NO_ONLY % ("derive_text")}
        if (not isinstance(derive_text_specific_cols_dict, dict)):
            return {"success": False, "message": tr_api.INVALID_DATATYPE_PARAMETER % ("text_types") + tr_api.VALID_DATATYPE % ("dict")}
        if (not isinstance(concepts_dict, dict)):
            return {"success": False, "message": tr_api.INVALID_DATATYPE_PARAMETER % ("phrases") + tr_api.VALID_DATATYPE % ("dict")}
        if (not isinstance(expressions_list, list)):
            return {"success": False, "message": tr_api.INVALID_DATATYPE_PARAMETER % ("expressions") + tr_api.VALID_DATATYPE % ("list")}
        
        extra_info = {}
        extra_info["misc_data"] = {}
        extra_info["misc_data_model"] = {}
        extra_info["model_data"] = {}
        extra_info["g_did_mid"] = g
        extra_info["outcome"] = outcome
        extra_info["misc_data"]["is_imputation_required"] = False

        if spark:
            return {"success": False, "message": "This version currently does not support the spark module for building models."}
            extra_info["spark"] = spark
            dtypes_list = train_data.dtypes
            dtype_df = pd.DataFrame(dtypes_list, columns=[g.VARIABLE_NAME, g.DATA_TYPE])
        else:
            dtype_df, ps_df = utility.get_smart_datatypes(train_data, extra_info)

        date_types = dtype_df.loc[dtype_df[g.DATA_TYPE]
                                    == g.DT_DATETIME][g.VARIABLE_NAME].tolist()
        cat_types = dtype_df.loc[dtype_df[g.DATA_TYPE]
                                   == g.DT_CATEGORICAL][g.VARIABLE_NAME].tolist()
        text_types = dtype_df.loc[dtype_df[g.DATA_TYPE]
                                    == g.DT_TEXT][g.VARIABLE_NAME].tolist()
        num_types = dtype_df.loc[dtype_df[g.DATA_TYPE]
                                   == g.DT_NUMERIC][g.VARIABLE_NAME].tolist()
           
        log.log_db(f"Data Type process has been finished")
        
        
       
        pdata_cat_cols_unique_list = dict()
        if spark:
            if outcome not in train_data.columns:
                return {"success": False, "message": tr_api.VALID_BUILD_OUTCOME.replace("this", "outcome")}
            data_correctness = spark_utils.check_numerical_columns(train_data, outcome)
            if not data_correctness:
                return {"success": False, "message": tr_api.VALID_SPARKDATAOBJECT}

            null_columns = [column for column in train_data.columns if train_data.filter(col(column).isNull()).count() > 0]
            if null_columns:
                return {"success": False, "message": tr_api.VALID_DATANULLOBJECT.replace("this", "train_data")}
            
            res_stats = spark_utils.get_statistics(train_data, num_types, cat_types, date_types, 
                                                   text_types, spark, null_handler=g.NA)
            extra_info["misc_data_model"][g.MIN_MAX] = None
            #no prefix suffix in spark end. processing done at client end.
            ps_df = None
            n = train_data.select(outcome).distinct().rdd.flatMap(lambda x: x).collect()
            if len(n) <= min(config.CATEGORICAL_UPPER_THRESHOLD, g.CATEGORICAL_LOWER_THRESHOLD):
                outcome_type = "categorical"
                extra_info["misc_data"]["outcome_labels"] = n
            else:
                outcome_type = "regression"
                
            
            for column in cat_types:
                # Get unique values for each column in PySpark
                unique_values = train_data.select(column).distinct().rdd.flatMap(lambda x: x).collect()
                pdata_cat_cols_unique_list[column] = unique_values
        else:
            if outcome not in list(train_data.columns):
                return {"success": False, "message": tr_api.VALID_BUILD_OUTCOME.replace("this", "outcome")}
            if train_data.isnull().values.any():
                return {"success": False, "message": tr_api.VALID_DATANULLOBJECT.replace("this", "train_data")}
            # Called in inform statistics
            train_data = utility.convert_data_types(train_data, cat_types, num_types, date_types)
            num_df = utility.get_statistics(train_data[num_types], g.DT_NUMERIC)
            cat_df = utility.get_statistics(train_data[cat_types], g.DT_CATEGORICAL)
            dt_df = utility.get_statistics(train_data[date_types], g.DT_DATETIME)
            text_df = utility.get_statistics(train_data[text_types], g.DT_TEXT)
            res_stats = pd.concat([num_df, cat_df, text_df, dt_df], axis=0, ignore_index=True)
            
            outcome_type = dtype_df.loc[dtype_df[g.VARIABLE_NAME] == outcome][g.DATA_TYPE].tolist()[0]
            pdata_cat_cols_unique_list = dict()
            for column in cat_types:
                pdata_cat_cols_unique_list[column] = train_data[column].unique().tolist()
                                 
        
        if outcome_type == "categorical":
            extra_info["misc_data"]["model_type"] = "CL"
            #extra_info["model_type"] = "CL"
            # Calculate the total number of unique classes
            if not spark:
                total_classes = train_data[outcome].nunique()

                # Get the value counts for each class
                value_counts = train_data[outcome].value_counts()

                # Check if any class has fewer data points than the total number of classes
                insufficient_classes = value_counts[value_counts < total_classes*config.CLASSIFICATION_CLASS_FACTOR]

                insufficient_classes_minpoints = value_counts[value_counts < config.CATEGORICAL_CLASS_MINPOINTS]

                if len(insufficient_classes)!=0 and len(insufficient_classes_minpoints)!=0:
                    return {"success": False, "message": "The Train_data does not have enough values in each class to build a model."}
        else:
            extra_info["misc_data"]["model_type"] = "PR"
        
        extra_info["model_type"] = extra_info["misc_data"]["model_type"]
        model_type = extra_info["model_type"]
        
        log.log_db(f"Data Statisitcs and Model type identification process has been finished")        
         
        extra_info["misc_data"][g.STAT] = res_stats
        extra_info["misc_data"][g.PRESUF_DF] = ps_df
        extra_info["misc_data"][g.PDATA_CAT_COLS_UNIQUE_LIST] = pdata_cat_cols_unique_list
        extra_info["misc_data"]["Data Type"] = dtype_df
        misc_data = extra_info["misc_data"]
        model_data = extra_info["model_data"]

        if misc_data[g.IMPUTATION_REQUIRED]:
            if not g.IS_IMPUTATION_DONE in misc_data:
                return {"success": False, "message": tr_api.DATA_HAS_MISSING_VALUES}
        
        if is_accelerated_required.lower() == "yes" and not spark:
            #For TS models, we directly build models
            if model_type == "TS":
                return {'success': False, 'message': tr_api.MODEL_BUILD_NOT_POSSIBLE}
                
            else:
                log.log_db(f"Data Acceleration module has been started")        

                vif_threshold = 50
                derived_predictors_threshold = 50
              
                #Remove dependent predictors according to the user"s command
                if remove_dependent_cmd.lower() == "yes":
                    ret_dict = build_model_helper.inform_removal_of_dependent_predictors(train_data, cmd="1", 
                                                                                         display=False, extra_info=extra_info)
                else:
                    ret_dict = build_model_helper.inform_removal_of_dependent_predictors(train_data, cmd="2", 
                                                                                         display=False, extra_info=extra_info)
                    
                #Saving the user"s options for numeric derived predictors if numeric columns are present
                if build_model_helper.datatype_col_present(train_data, extra_info=extra_info):
                    if derive_numeric_cmd.lower() == "yes":
                        ret_dict = build_model_helper.ask_for_derived_predictors(cmd="1", display=False, extra_info=extra_info)
                        is_derived_numeric_possible, derived_df = build_model_helper.derive_numeric_for_api(expressions_list,
                                                                                                            extra_info=extra_info)
                        if not is_derived_numeric_possible:
                            ret_dict = build_model_helper.ask_for_derived_predictors(train_data, cmd="2", 
                                                                                     display=False, extra_info=extra_info)
                    else:
                        ret_dict = build_model_helper.ask_for_derived_predictors(train_data, cmd="2", 
                                                                                 display=False, extra_info=extra_info)
                        #return ret_dict, extra_info
                else:
                    ret_dict = build_model_helper.ask_for_derived_predictors(train_data, cmd="2", 
                                                                             display=False, extra_info=extra_info)
                
                #Saving the user"s options for text derived predictors if text columns are present
#                 if build_model_helper.datatype_col_present(df, g.DT_TEXT, extra_info=extra_info):
#                     if derive_text_cmd.lower() == "yes":
#                         ret_dict = build_model_helper.ask_for_derived_text_predictors(cmd="1", 
#                                                                                       display=False, extra_info=extra_info)
#                         is_derived_text_possible, derived_df = build_model_helper.derive_text_for_api(concepts_dict, derive_text_specific_cols_dict, extra_info=extra_info)
#                         if not is_derived_text_possible:
#                             ret_dict = build_model_helper.ask_for_derived_text_predictors(cmd="2", 
#                                                                                           display=False, extra_info=extra_info)
#                     else:
#                         ret_dict = build_model_helper.ask_for_derived_text_predictors( cmd="2", 
#                                                                                       display=False, extra_info=extra_info)
#                 else:
#                     ret_dict = build_model_helper.ask_for_derived_text_predictors(cmd='2', 
#                                                                                   display=False, extra_info=extra_info)
                
                log.log_db(f"Feature Selection and extraction")        

                #Feature extraction
                is_feature_selection_possible, selected_features_list, \
                selected_score_list, extra_info = build_model_helper.feature_extraction_for_api(train_data, 
                                                                                                extra_info=extra_info)
                
                if not is_feature_selection_possible:
                    return {'success': False, 
                            'message': 'Feature selection is not possible as there is no numeric columns left after encoding.'}
                
                log.log_db(f"Feature Selection and extraction is processed") 
            
                #return extra_info, status_code
                #Build Models
                var_type = misc_data[g.DATA_TYPE]
                cat_types = var_type.loc[var_type[g.DATA_TYPE] == g.DT_CATEGORICAL][g.VARIABLE_NAME].tolist()
                if extra_info["outcome"] in cat_types:
                    cat_types.remove(extra_info["outcome"])
                train_data = utility.create_dummy_features(train_data, cat_types)
                added_columns, train_data = utility.get_date_time_features(train_data, date_types)

                log.log_db(f"Modelling Initialized") 

                if not g.API_FLEXIBILITY:
                    performance_dict = build_model_helper.build_model_show_core_predictors(train_data, display=False, 
                                                                                           extra_info=extra_info)
                    try:
                        performance_dict = json.loads(performance_dict.get_data())
                    except Exception as e:
                        pass
                    if performance_dict is None:
                        return {"success": False, "message": tr_api.INTERNAL_SERVER_ERROR}

                else:
                    return {'success': False, 'message': tr_api.MODEL_BUILD_NOT_POSSIBLE}
                
                log.log_db(f"Global Feature importance")
                global_importance_df = build_model_helper.show_core_predictors(cmd="", display=True, return_df=True, extra_info=extra_info)
                global_importance_dict_to_be_returned = dict()
                global_importance_dict_to_be_returned["data"] = global_importance_df.values.tolist()
                global_importance_dict_to_be_returned["columns"] = global_importance_df.columns.tolist()
                global_importance = pd.DataFrame(columns=global_importance_dict_to_be_returned['columns'], 
                                                    data=global_importance_dict_to_be_returned['data'])
                log.log_db(f"Extra Info derivation and encription")
                #Return Model scores and global importance values
                output_data = api_utils.output_extra_info(extra_info)
                output_data = api_utils.encrypt_dict(output_data)
                log.log_db(f"All steps Completed")
                performance = utility.decode_json_dict(performance_dict[g.RIGHT][g.TABLE])
                pred_performance_df = pd.DataFrame(columns=performance['columns'], data=performance['data'])
                return {"success": True, "message": tr_api.MODEL_BUILT, 
                        "model_performance": pred_performance_df, \
                        "global_importance": global_importance, "model_info": output_data}
        elif spark:
            log.log_db(f"Spark Modelling Initialized")

            performance_dict = build_model_helper.build_model_show_core_predictors(train_data, extra_info=extra_info)
            
            if performance_dict is None:
                return {"success": False, "message": tr_api.INTERNAL_SERVER_ERROR}

            log.log_db(f"Spark Global Feature importance")
            
            global_importance_df = build_model_helper.show_core_predictors(cmd="", display=True, return_df=True, extra_info=extra_info)
            global_importance_dict_to_be_returned = dict()
            global_importance_dict_to_be_returned["data"] = global_importance_df.values.tolist()
            global_importance_dict_to_be_returned["columns"] = global_importance_df.columns.tolist()
            global_importance = pd.DataFrame(columns=global_importance_dict_to_be_returned['columns'], 
                                                    data=global_importance_dict_to_be_returned['data'])
            log.log_db(f"Extra info derivation and encription")
            #Return Model scores and global importance values
            if spark:
                spark_module = extra_info["model_data"]["Consolidated Metrics"]
                spark_info = api_utils.output_extra_info(extra_info, spark=spark)
                spark_encript = api_utils.encrypt_dict(spark_info)
                output_data = {"spark_info" : spark_encript,
                             "spark_module" : spark_module}
            
            log.log_db(f"All Spark steps Completed")

            #output_data = extra_info
            performance = utility.decode_json_dict(performance_dict[g.RIGHT][g.TABLE])
            pred_performance_df = pd.DataFrame(columns=performance['columns'], data=performance['data'])
            return {"success": True, "message": tr_api.MODEL_BUILT, 
                    "model_performance": pred_performance_df, \
                    "global_importance": global_importance, "model_info": output_data}
    except Exception as e:
        log.log_db(("Exception in ez_model", e))
        log.log_db(traceback.print_exc())
        return {"success": False, "message": tr_api.INTERNAL_SERVER_ERROR}


def ez_predict(test_data, model_info, options={}):
    """
    Perform predictions on the provided test data using the model parameters generated by ez_build_model.

    Parameters:
        - **test_data** (`DataFrame` or `str`): The dataset to be evaluated. It must have the same features as the dataset used for training.
        - **model_info** (`Bytes`): Contains the encrypted or unencrypted details about the trained model and its environment.
        - **options** (`dict`): A dictionary of configuration options for model initialization and prediction. Supported keys include:
            
            - **model** (`str`, optional): Specifies the model to be used for prediction. If not provided, the default model from `model_info` is used.
            - **confidence_score** (`bool`, optional): Default is `False`. If `True`, the function provides a confidence score for classification models.
            - **spark_session** (`SparkSession` or `None`, optional): If provided, a Spark session will be used for distributed computation. If `None`, standard computation is used.
            - **spark_model** (`model` or `pipeline`, optional): If the model is saved and `spark_session` is provided, the trained Spark model or pipeline should be loaded and passed here.

    Returns:
        - **dict**: A dictionary containing the result of the evaluation. The dictionary contains the following keys:
            
            - **"success"** (`bool`): Indicates whether the operation was successful.
            - **"message"** (`str`): A message containing either an error or informational details.
            
            If successful, the dictionary also contains:
            
            - **"pred_df"** (`DataFrame`): A DataFrame containing the predictions for the test dataset.



    Example:
        .. code-block:: python

            import pandas as pd
            import joblib
            from eazyml import ez_predict

            # Load test data.
            test_file_path = "path_to_your_test_data.csv"
            test_data = pd.read_csv(test_file_path)

            # Load output from ez_build_model. This should be the file where model information is stored.
            build_model_response_path = 'model_response.joblib'
            build_model_response = joblib.load(build_model_response_path)
            model_info = build_model_response["model_info"]

            # Choose the model for prediction from the key "model_performance" in the build_model_response object above. The default model is the top-performing model if no value is provided.
            pred_options = {"model": "Random Forest with Information Gain"}

            # Call the eazyml function to predict
            pred_response = ez_predict(test_data, model_info, options=pred_options)

            # prediction response is a dictionary object with following keys.
            # print(pred_response.keys())
            # dict_keys(['success', 'message', 'pred_df'])
            
    """    
    try:
        log.log_db(f"Initialze ez_predict")

        global g
        #Check for valid keys in the options dict
        for key in options:
            if key not in tr_api.EZ_PREDICT_OPTIONS_KEYS_LIST:
                return {"success": False, "message": tr_api.INVALID_KEY % (key)}

        if isinstance(test_data, str):
            if not os.path.exists(test_data):
                return {"success": False,"message": "test_data provided path does not exist."}
            else:
                test_file_path = test_data
                if "spark_session" in options:
                    spark = options["spark_session"]
                else:
                    spark = None
                if spark:
                    try:
                        spark_version = spark.version
                    except:
                        return {"success": False, "message": tr_api.SPARK_SESSION}
                    test_data = spark_utils.get_df_spark(test_file_path, spark)
                else:
                    test_data = upload_utils.get_df(test_file_path)
            if test_data is None:
                return {"success": False, "message": tr_api.VALID_DATAFILEPATH.replace("this", "test_data")} 
        elif not isinstance(test_data, (pd.DataFrame, DataFrame)):
            return {"success": False, "message": tr_api.VALID_DATAOBJECT.replace("this", "test_data")}
        elif isinstance(test_data, DataFrame):
            if "spark_session" in options:
                spark = options["spark_session"]
            else:
                spark = None
            if spark:
                try:
                    spark_version = spark.version
                except:
                    return {"success": False, "message": tr_api.SPARK_SESSION}

        if not isinstance(options, dict):
            return {"success": False, "message": tr_api.VALID_DATATYPE_DICT.replace("this", "options")}

        if "spark_session" in options:
            log.log_db(f"Spark Predict")
            spark = options["spark_session"]
        else: 
            spark = None

        if spark: 
            return {"success": False, "message": "This version currently does not support the spark module for building models."}
            if isinstance(model_info,bytes):
                try:
                    model_info = api_utils.decrypt_dict(model_info)
                except Exception as e:
                    log.log_db(f"model info decript: {e}")
                    return {"success": False, "message": tr_api.VALID_INFO.replace("this", "Model info")}
                if "spark_model" not in options:
                    return {"success": False, "message": tr_api.VALID_SPARK_MODEL}
                else:
                    spark_model = options["spark_model"]
                    try:
                        total_stages = spark_model.stages
                    except:
                        return {"success": False, "message": tr_api.VALID_SPARK_MODEL}
                    try:
                        model_info["misc_data"]["Data Type"] =  pd.DataFrame(model_info["misc_data"]["Data Type"])
                        model_info["model_data"]["Consolidated Metrics"] =  pd.DataFrame(model_info["model_data"]["Consolidated Metrics"])
                        model_info["model_data"]["metrics"] =  pd.DataFrame(model_info["model_data"]["metrics"])
                        # Apply the function and create two columns: 'Models' and 'Flag'
                        model_info["model_data"]["Consolidated Metrics"][['Models', 'Flag']] = \
                            model_info["model_data"]["Consolidated Metrics"]['Models'].apply(
                                lambda row: pd.Series(api_utils.spark_model_name_map(row, spark_model))
                            )
                        model = model_info["model_data"]["Consolidated Metrics"][model_info["model_data"]["Consolidated Metrics"]["Flag"]==True]["Model"].to_list()[0]
                    except:
                        return {"success": False, "message": "Please provide valid spark_model"}

            elif isinstance(model_info, dict):
                if "spark_info" in model_info.keys():
                    if isinstance(model_info["spark_info"],bytes):
                        try:
                            model_info["spark_info"] = api_utils.decrypt_dict(model_info["spark_info"])
                        except Exception as e:
                            log.log_db(f"spark info decript: {e}")
                            return {"success": False, "message": tr_api.VALID_INFO.replace("this", "spark info")}
                    model_info["spark_info"]["model_data"]["metrics"] =  pd.DataFrame(model_info["spark_info"]["model_data"]["metrics"])
                    model_info["spark_info"]["misc_data"]["Data Type"] =  pd.DataFrame(model_info["spark_info"]["misc_data"]["Data Type"])
                    model_info["spark_info"]["model_data"]["Consolidated Metrics"] = pd.DataFrame(model_info["spark_module"])
                    model_info = model_info["spark_info"]
                    if "model" in options:
                        model = options["model"]
                    else:
                        model = model_info["model_data"]["metrics"]["Model"][0]
                else:
                    return {"success": False, "message": tr_api.VALID_SPARK_INFO.replace("this", "spark info")}
            else:
                return {"success": False, "message": tr_api.VALID_SPARK_INFO.replace("this", "spark info")}
                
        else:
            if isinstance(model_info,bytes):
                try:
                    model_info = api_utils.decrypt_dict(model_info)
                except Exception as e:
                    log.log_db(f"model info decript: {e}")
                    return {"success": False, "message": tr_api.VALID_INFO.replace("this", "Model info")}
        try:
            model_info["model_data"]["Consolidated Metrics"] =  pd.DataFrame(model_info["model_data"]["Consolidated Metrics"])
            model_info["model_data"]["metrics"] =  pd.DataFrame(model_info["model_data"]["metrics"])

            model_info["misc_data"]["prefix_suffix_dataframe"] =  pd.DataFrame(model_info["misc_data"]["prefix_suffix_dataframe"])
            model_info["misc_data"]["statistics"] =  pd.DataFrame(model_info["misc_data"]["statistics"])
            model_info["misc_data"]["Data Type"] =  pd.DataFrame(model_info["misc_data"]["Data Type"])
        except Exception as e:
            log.log_db(f"model info convert to dataframe could already be in dataframe structure.error below")
            log.log_db(f"model info convert to dataframe: {e}")
                
        log.log_db(f"Model info provided has been decripted")
        
        model_info_keys = ['misc_data', 'misc_data_model', 'model_data', 'model_type', 'outcome']
        for each_para in model_info_keys:
            if each_para not in model_info.keys():
                return {"success": False, "message": tr_api.VALID_EXTRA_INFO.replace("this", "model_info")}

        if "confidence_score" in options:
            confidence_score = options["confidence_score"]
        else:
            confidence_score = False
           
        if "class_probability" in options:
            class_probability = options["class_probability"]
        else:
            class_probability = False

        model_info["g_did_mid"] = g
        model_data = model_info["model_data"]
        models_list = model_data[g.METRICS]["Model"].values.tolist()
        model_info["spark"] = spark
        
        if not spark:
            if "model" in options:
                model = options["model"]
                if model not in models_list:
                    return {"success": False, "message": tr_api.VALID_MODEL_PARAM}
            else:
                model = model_info["model_data"]["metrics"]["Model"][0]

        model_info["model_data"]["predict_model"] = model
        difference = test_helper.check_if_test_data_is_consistent(test_data, extra_info=model_info)
        log.log_db(f"test_data found consistent compared to train_data")
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
                return {"success": False,"message": tr_api.INCONSISTENT_DATASET}
            
        log.log_db(f"Processing test data")
        test_data, col_name = test_helper.process_test_data(test_data, extra_info=model_info)
        if test_data is None:
            return {"success": False,"message": tr.TRAIN_TEST_COLUMN_MISMATCH % col_name}
        elif spark:
            if test_data.isEmpty():
                return {"success": False, "message": tr.INVALID_VALUE}
        elif test_data.empty:
            return {"success": False, "message": tr.INVALID_VALUE}
        log.log_db(f"Prediction initiated")
        res = test_helper.show_results(test_data, display=True, extra_info=model_info, confidence_score=confidence_score,
                                      class_probability=class_probability)
        
        if isinstance(res, (pd.DataFrame, DataFrame)):
            return {"success": True, "message": "Prediction has been completed", "pred_df":res}
        elif res == tr_api.VALID_MODEL_PARAMETER:
            return {"success": False, "message": tr_api.VALID_MODEL_PARAMETER}
        else:
            return {"success": False, "message": tr_api.INTERNAL_SERVER_ERROR}
    except Exception as e:
        log.log_db(("Exception in ez_predict", e))
        log.log_db(traceback.print_exc())
        return {"success": False, "message": tr_api.INTERNAL_SERVER_ERROR}


def ez_display_json(resp):
    """
    Function to display formatted json
    """
    return display_json(resp)


def ez_display_df(resp):
    """
    Function to display formatted dataframe
    """
    return display_df(resp)


def ez_display_md(resp):
    """
    Function to display formatted markdown
    """
    return display_md(resp)
