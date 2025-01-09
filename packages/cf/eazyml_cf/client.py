"""
EazyML revolutionizes machine learning by introducing counterfactual inference,
automating the process of identifying optimal changes to variables that shift
outcomes from unfavorable to favorable. This approach overcomes the limitations
of manual "what-if" analysis, enabling models to provide actionable, prescriptive
insights alongside their predictions.
"""
import os

from .src.cfr_helper import (
            cf_inference,
            scikit_feature_selection,
            scikit_model_building
)

from .license.license import (
        validate_license,
        init_eazyml,
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
def ez_cf_inference(train_file, test_file, outcome, config, 
                            selected_columns, model_info, test_record_idx):
   """
   This function performs counterfactual inference on a test record. It calculates the probability 
   of an unfavorable outcome for a given test record and finds an optimal point to minimize this probability.

   Parameters:
      - **train_file** (str):
         The file path to the training dataset in Excel format.
      - **test_file** (str):
         The file path to the testing dataset in Excel format.
      - **outcome** (str):
         The name of the target column (dependent variable).
      - **config** (dict): A configuration dictionary containing relevant settings for counterfactual inference, including:
      
         .. code-block:: python

            config = {
                  "unfavorable_outcome": "...",
                  "invariant_features": [...]
            }

         **Key Fields**:
            - `"unfavorable_outcome"`: The value representing an unfavorable outcome.
            - `"invariant_features"`: A list of features that should remain unchanged during counterfactual optimization.
            - Other relevant configuration parameters.

      - **selected_columns** (list):
         A list of selected feature column names used in the model.
      - **model_info** (dict):
         A dictionary containing the trained model (``clf``) and other related model information.
      - **test_record_idx** (int):
         The index of the test record on which counterfactual inference is performed.

   Returns :
      - **Dictionary with Fields** :
         - **success** (bool): Indicates whether the inference was successful.
         - **summary** (dict): Summary of the probabilities for the test and optimal points.
            - "*Actual Test Point Unfavorable Outcome Probability*" (float): Probability of an unfavorable outcome for the test record.
            - "*Optimal Point Unfavorable Outcome Probability*" (float): Probability of an unfavorable outcome for the optimal point.
         - **details** (dict): Additional details about the feature value changes by the inference process.

   Example :
      .. code-block:: python

         ez_cf_inference(
            train_file = 'train/file/path.csv',
            test_file = 'test/file/path.csv',
            outcome = 'outcome column name',
            config= {
               "unfavorable_outcome" : 1,
               "lower_quantile" : 0.01,
               "upper_quantile" : 0.99,
               "p" : 40,
               "M" : 2,
               "N" : 10000,
               "tolerable_error_threshold" : 0.1
            },
            selected_columns = 'List of selected input features',
            model_info = 'dictionary of model information'
            test_record_idx = 'single or multiple testdata id or None'
         )
   """
   return cf_inference(train_file, test_file, outcome, config, 
                            selected_columns, model_info, test_record_idx)


@validate_license
def sk_feature_selection(train_file, outcome, config):
   """
   This function performs feature selection from a training dataset by excluding specific columns and the target outcome column.

   Parameters :
      - **train_file** (str):  
         The file path to the training dataset in Excel format.

      - **outcome** (str):  
         The name of the target column (dependent variable).

      - **config** (dict): A configuration dictionary containing:  

         .. code-block:: python

            config = {
               "discard_columns": [...]
            }

         **Key Fields**:
            - `"discard_columns"`: A list of column names to exclude from feature selection.

       
   Returns:
      - **selected_columns** (list):
         A list of column names after the feature selection.
   
   Example :
      .. code-block:: python

         sk_feature_selection(
            train_file = 'train/file/path.csv',
            outcome = 'outcome column name',
            config= {
               "discard_columns" : ['id', 'unnamed'],
            }
         )
   """
   return scikit_feature_selection(train_file, outcome, config)

@validate_license
def sk_model_building(train_file, test_file, outcome, selected_columns, config):
   """
   This function builds a machine learning model using a specified training dataset.

   Parameters:
      - **train_file** (str):
         The file path to the training dataset in Excel format.
      - **outcome** (str):
         The name of the target column (dependent variable).
      - **selected_columns** (list):
         A list of selected feature column names for training the model.
      - **config** (dict): A configuration dictionary containing:

         .. code-block:: python

            config = {
               "sklearn_classifier": "..."
            }

         **Key Fields**:
            - `"sklearn_classifier"`: Sklearn Classifier to use for model building.

   Returns:
      - `clf`: The trained model object.
      - `model_info (dict)`: A dictionary containing:

   Example :
      .. code-block:: python

         sk_feature_selection(
            train_file = 'train/file/path.csv',
            test_file = 'test/file/path.csv',
            outcome = 'outcome column name',
            selected_columns = 'List of selected input features',
            config= {
               "unfavorable_outcome" : 1,
               "sklearn_classifier" : 'Gradient Boosting'
            }
         )
   """
   return scikit_model_building(train_file, test_file, outcome, selected_columns, config)

