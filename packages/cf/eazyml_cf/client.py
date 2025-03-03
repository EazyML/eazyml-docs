"""
EazyML revolutionizes machine learning by introducing counterfactual inference,
automating the process of identifying optimal changes to variables that shift
outcomes from unfavorable to favorable. This approach overcomes the limitations
of manual "what-if" analysis, enabling models to provide actionable, prescriptive
insights alongside their predictions.
"""
import os
import pandas as pd

from .src.ez_counterfactual_inference import (
            cf_inference
)

from .license.license import (
        validate_license,
        init_eazyml,
)

from .src.utils import CustomError, BaseEstimator

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

   
def ez_cf_inference(test_data, outcome, selected_features, 
                    model_info, options):
    """
    Perform counterfactual inference on a test record to optimally transition from an unfavorable outcome to a favorable one.
    This function identifies optimal changes to selected features that can improve the predicted outcome, based on a given model's inference.

    Args:
        - **test_data** (`DataFrame` or `str`): A pandas DataFrame containing a single test record, including the target outcome and necessary features for inference. The DataFrame must have the same feature set as the model's training data.
        - **outcome** (`str`): The name of the target column (dependent variable) representing the outcome to be predicted. This can be a binary label (e.g., success/failure, 1/0) or continuous values.
        - **selected_features** (`list of str`): A list of feature names used by the model for making predictions. These features should match the model's input expectations.
        - **model_info** (`bytes` or `model object`) : A serialized model or model object containing the predictive model used for inference. This parameter supports two types of inputs:
            
            - For **EazyML modeling**, the model_info should be provided as a bytes object representing the serialized model.
            - For **Custom modeling** (e.g., sklearn-based models), the model_info should be the model object directly passed to the function.
        
        - **options** (`dict`): A dictionary specifying configuration settings for counterfactual inference. The following keys are supported:
            
            - `variants` (list of str): List of feature names that can be modified during the inference process.
            - `outcome_ordinality` (str or int): Specifies the target outcome direction or value.
            
                Regression problems:
                    - `"maximize"`: The function attempts to increase the outcome value to the most favorable possible result.
                    - `"minimize"`: The function attempts to decrease the outcome value to the least unfavorable possible result.
                
                Classification problems:
                    - Binary classification:
                        - `0/1`: The function tries to modify the features to reach the target value of 0/1 (integer-based outcome).
                        - `"preferred_outcome_string"`: The function tries to modify the features to reach the target outcome as a string.
                    - Multi-class classification:
                        - `0/1/2/...`: The function tries to modify the features to reach the target value of 0/1/2... (any integer outcome value in the outcome column).
                        - `"preferred_outcome_string"`: The function tries to modify the features to reach the target outcome as a string.

    Returns:
        - A dictionary containing the following fields:
            
            - `success` (bool): Indicates whether a favorable counterfactual solution was successfully found.
            - `message` (str): Describes the outcome of the inference process.
            - `summary` (dict): Provides a summary of the inference process, including:
            
                - `Actual Outcome` (str): The outcome of the test record before applying counterfactual inference.
                - `Optimal Outcome` (str): The derived outcome after applying counterfactual inference.
                - `Improvement in Probability` (float, optional): Difference in probability between the actual and optimal outcomes (applicable only for classification problems).
        - A DataFrame showing the feature values before and after counterfactual inference, along with the degree of change for each feature.

    Examples:
        **EazyML modeling**
        
        .. code-block:: python
            
            result, optimal_transition_df = ez_cf_inference(
                test_data=test_data,
                outcome='outcome_column_name',
                selected_features=['feature1', 'feature2', 'feature3'],
                model_info=b"acacxaca",
                options={
                    "variant_type": "static",
                    "variants": ["feature1", "feature2"],
                    "outcome_ordinality": "desired_favorable_outcome"
                }
            )

        **Custom modeling**
        
        .. code-block:: python
            
            result, optimal_transition_df = ez_cf_inference(
                test_data=test_data,
                outcome='outcome_column_name',
                selected_features=['feature1', 'feature2', 'feature3'],
                model_info=model_object,
                options={
                    "variant_type": "static",
                    "variants": ["feature1", "feature2"],
                    "outcome_ordinality": "desired_favorable_outcome",
                    "preprocessor": preprocessor_object
                }
            )

        **Return Values**
        
        .. code-block:: python
        
            result = {
                    "success": True,
                    "message": "Optimal transition found",
                    "summary": {
                        "Actual Outcome": "0",
                        "Optimal Outcome": "1",
                        "Improvement in Probability": 0.416
                    }
                }
        
            optimal_transition_df = pd.DataFrame({
                    "Feature": ["Years In Business", "Deposit Sum 30 MA", "Total Postings", "Deposit Count 90"],
                    "Actual": [2.29, 32554.0, 13.0, 13.0],
                    "Optimal": [2.29, 29203.22, 15.6, 10.4],
                    "Percentage Change": [0.0, -10.3, 20.0, -20.0],
                    "Absolute Change": [0.0, -3350.78, 2.6, -2.6]
                })
    
    Note:
        - In multi-class classification, if the inference doesn't result in the target outcome, it tries to reach the next best outcome and so on. For example, if the actual outcome is 0 and the target outcome is 3, the inference will attempt to reach 3, and if not possible, then 2, then 1.
        - `train_data` (str or DataFrame): Training data used to fit the model.
        - `preprocessor` (object, optional): Preprocessing object used to transform input data during inference (required for custom models). The preprocessor should be a bundled class containing methods that internally transform and inverse transform the data as required.
            
            Regression:
                - `transform`: Transforms the input data to the format expected by the model.
                - `inverse_transform_outcome`: Reverses the outcome transformation to obtain the original scale.
            
            Classification:
                - `transform`: Transforms the input data to the format expected by the model.
                - `label_encoder`: Encodes and decodes class labels for categorical outcomes.
    """
    

    try:
        # Validate inputs
        validate_input(test_data, outcome, selected_features, model_info, options)

        # Perform the inference if inputs are valid
        return cf_inference(test_data, outcome, selected_features, 
                            model_info, options)
    except CustomError as e:
        # Handle the error as needed
        print(f"Custom Error encountered: {e}")
        return None     

def validate_input(test_record_df, outcome, selected_features, model_info, options):
    """
    Validates the input parameters before processing.

    Parameters:
        test_record_df (pd.DataFrame): A single-row DataFrame containing the test record.
        outcome (str): The target variable name.
        selected_features (list): List of selected feature names.
        model_info (bytes/object): String containing model-related metadata or model object.
        options (dict): Dictionary containing optional parameters for processing.

    Raises:
        CustomError: If any validation fails.
    """

    # Validate test_record_df
    if not isinstance(test_record_df, pd.DataFrame):
        raise CustomError("test_record_df must be a pandas DataFrame.")
    if test_record_df.shape[0] != 1:
        raise CustomError("test_record_df must contain exactly one row.")

    # Validate outcome
    if not isinstance(outcome, str):
        raise CustomError("outcome must be a string.")

    # Validate selected_features
    if not isinstance(selected_features, list):
        raise CustomError("selected_features must be a list of strings.")
    if not all(isinstance(feature, str) for feature in selected_features):
        raise CustomError("All elements in selected_features must be strings.")
    if not set(selected_features).issubset(test_record_df.columns):
        raise CustomError("Some elements in selected_features are not present in test_record_df.")

    # Validate model_info
    if not isinstance(model_info, (bytes, BaseEstimator)):
        raise CustomError("model_info must be bytes, or scikit-learn model object.")

    # Validate options
    if not isinstance(options, dict):
        raise CustomError("options must be a dictionary.")

    # Define expected options and validation rules
    validations = {
        "variants": lambda x: isinstance(x, list) and all(isinstance(v, str) for v in x),
        "outcome_ordinality": lambda x: isinstance(x, str),
        "qualifier": lambda x: isinstance(x, str),
        "lower_quantile": lambda x: isinstance(x, float) and 0 <= x <= 1,
        "upper_quantile": lambda x: isinstance(x, float) and 0 <= x <= 1,
        "vicinity_region_percentage": lambda x: isinstance(x, int) and 0 <= x <= 100,
        "N": lambda x: isinstance(x, int) and x > 0,
        "tolerable_error_threshold": lambda x: isinstance(x, float) and 0 <= x <= 1,
        "variant_type": lambda x: isinstance(x, str) and x in {"static", "dynamic"},
        "continuous_percent": lambda x: isinstance(x, int) and 0 <= x <= 100,
        "categorical_percent": lambda x: isinstance(x, int) and 0 <= x <= 100,
        "global_range_for_continuous": lambda x: isinstance(x, bool),
        "best_among_local_optimums": lambda x: isinstance(x, bool),
        "no_of_random_initial_seed": lambda x: isinstance(x, int) and x > 0
    }

    # Ensure required options exist
    missing_keys = [key for key in ["variants", "outcome_ordinality"] if key not in options]
    if missing_keys:
        raise CustomError(f"Missing required options: {', '.join(missing_keys)}")

    # Validate all provided options
    for key, check in validations.items():
        if key in options and not check(options[key]):
            raise CustomError(f"Invalid format for '{key}': {options[key]}")

    # Check if variants are a subset of selected_features
    if not set(options["variants"]).issubset(set(selected_features)):
        raise CustomError("Some elements in 'variants' are not present in 'selected_features'.")

    return True  # If all validations pass
