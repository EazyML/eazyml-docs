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

