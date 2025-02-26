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

from .src.utils import CustomError

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

   
@validate_license
def ez_cf_inference(test_data, outcome, selected_features, 
                    model_info, options):
    """
    Perform counterfactual inference on a test record to optimally transition from an unfavorable outcome to a favorable one.


    Parameters:
       - **test_data** (DataFrame):
          A pandas DataFrame containing a single test record. The DataFrame should include predictions 
          for the outcome and the necessary features for inference.
          
       - **outcome** (str):
          The name of the target column (dependent variable) representing the outcome to be predicted 
          (e.g., a binary label like success/failure, 1/0, or continuous values).
       
       - **selected_features** (list of str):
          A list of features used by the actual model for making predictions.
       
       - **model_info** (dict):
          A dictionary containing the trained model and associated model information (e.g., the model object 
          and any necessary pre-processing steps).
       
       - **options** (dict):
          A dictionary of configuration settings for counterfactual inference, which may include:
          
          .. code-block:: python

             options = {
                 "variant_type": "static",  # Defines the type of variant adjustments (e.g., 'static' for fixed constraints)
                 "variants": ["list of features that can change"],  # List of feature names that can be modified during inference
                 "outcome_ordinality": "desired_outcome",  # Specifies the target outcome direction (e.g., 'maximize', 'minimize', or a specific value like 1/0).
             }

    Returns:
       - **dict**: A dictionary with the following fields:
          - **success** (bool): A flag indicating whether the counterfactual inference was successful.
          - **message** (str): A message describing whether an optimal transition was found or not.
          - **summary** (dict): A summary of the probabilities for the test record and the optimal point:
             - `"Actual Outcome"` (str): The actual outcome for the test record before the inference process.
             - `"Optimal Outcome"` (str): The optimally derived outcome after the counterfactual inference process.
             - `"Improvement in Probability"` (float): The difference in probability between the actual and optimal outcomes (only applicable for classification problems).
       - **DataFrame** (DataFrame): A detailed breakdown of the feature values and changes made during the inference process, 
           showing how the selected features were adjusted to arrive at the optimal outcome.

    Example:
       .. code-block:: python

          ez_cf_inference(
             test_record_df = test_data, 
             outcome = 'outcome_column_name',
             selected_features = ['feature1', 'feature2', 'feature3'],
             model_info = {'model_info': 'model_info'},
             options = {
                 "variant_type": "static",
                 "variants": ["feature1", "feature2"],
                 "outcome_ordinality": "desired_favorable_outcome"
             }
          )
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
        model_info (dict): Dictionary containing model-related metadata.
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
    if not isinstance(model_info, dict):
        raise CustomError("model_info must be a dictionary.")

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
