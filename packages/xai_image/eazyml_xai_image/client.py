"""
This package focuses on segmentation prediction, explainability, active learning and online learning for image dataset.

Active learning focuses on reducing the amount of labeled data required to train
the model while maximizing performance, making it particularly useful when labeling
data is expensive or time-consuming. By prioritizing uncertain or diverse examples,
active learning accelerates model improvement and enhances efficiency.

Online learning is a machine learning approach where models are trained incrementally
as data becomes available, rather than using a fixed, pre-existing dataset.
This method is well-suited for dynamic environments, enabling real-time updates
and adaptability to new patterns or changes in data streams.
"""
import os
from .globals.transparency_api import (
    INVALID_KEY, ALL_STR_PARAM,
    EXPLANATION_SUCCESS, INTERNAL_SERVER_ERROR,
    
)

from .globals.transparency_app import (
    EZ_IMAGE_EXPLAIN_OPTIONS_KEYS_LIST,
    EZ_IMAGE_SUPPORTED_TYPES, EZ_IMAGE_PRED_INPUT_FORMATS,
    EZ_IMAGE_DATA_PATH_FORMAT, EZ_IMAGE_XAI_STRATEGY,
    EZ_IMAGE_SUPPORTED_MODEL_FORMATS,
    EZ_IMAGE_EVALUATE_OPTIONS_KEYS_LIST,
    EZ_IMAGE_SCORE_STRATEGY, EZ_IMAGE_ACTIVE_OPTIONS_KEYS_LIST,
    EZ_IMAGE_AL_STRATEGY, EZ_IMAGE_ONLINE_OPTIONS_KEYS_LIST,
    EZ_IMAGE_OL_STRATEGY, EZ_IMAGE_TR_STRATEGY
)

from .src import exai
from .src import exai_main_image
import traceback

from .src.helper import ez_modify_options_seg_model
from .src.helper import (
            return_response, check_radio_parameter,
            is_list, is_string,
            check_path_parameter, check_if_positive_integer,
            check_list_of_path_parameter
    )


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


def ez_xai_image_explain(filename,
                         model_path,
                         predicted_filename,
                         options=None):
    """
    This API provides confidence scores and image explanations for model predictions.
    It can process a single image or multiple images, returning explanations for all predictions.

    Parameters :
        - **filename** (`str` | `list[str]`): Absolute path(s) of the image(s).
        - **model_path** (`str`): Absolute path to the model used for predictions.
        - **predicted_filename** (`str` | `list[str]`): Absolute path(s) of the predicted output(s).
        - **options** (`dict`): Configuration for explaining predictions. Example:

        .. code-block:: python

            options = {
                "training_data_path": "...",
                "score_strategy": "weighted-moments",
                "xai_strategy": "gradcam",
                "xai_image_path": "...",
                "gradcam_layer": "layer_name",
                "model_num": "1",
                "required_functions": {...}
            }

        - **Key Fields**:
            - `training_data_path`: Path to training data (absolute path; `.csv` format; columns: `inputs`, `labels`).
            - `score_strategy` (default: `"weighted-moments"`): Strategy for explainability score calculation.
            - `xai_strategy` (default: `"gradcam"`): Strategy for image explanation (e.g., `"gradcam"`, `"image-lime"`).
            - `xai_image_path`: Filename(s) for saving the explainability images.
            - `gradcam_layer`: Layer name for gradcam-based explanation.
            - `model_num`: Model ID if `score_strategy` is `"trainable-*"`.

    Returns :
        **Dictionary with Fields** :
            - `success` (`bool`): Indicates if the explanation generation was successful.
            - `message` (`str`): Describes the success or failure of the operation.
            - `explanations` (`list, optional`): The generated explanations (if successful) contains the explanation string or a local importance dataframe.
            - `confidence` (`float`): Explainability score for the prediction.

        .. code-block:: python

            {
                "success": <True|False>,
                "explanations": {
                    "explanation": "path/to/explainability/image",
                    "confidence": 95.4
                }
            }

    Example:
        .. code-block:: python

            ez_xai_image_explain(
                filename='path/to/image/image',
                model_path='path/to/model.h5',
                predicted_filename='path/to/prediction.csv/npy',
                options = {"score_strategy": "weighted-moments"}
            )
    """
    try:
        
        options = ez_modify_options_seg_model(options)

        # check if all the keys in options are expected.
        # There are no unexpected in options.
        for key in options:
            if key not in EZ_IMAGE_EXPLAIN_OPTIONS_KEYS_LIST:
                return {
                    "success": False,
                    "message": INVALID_KEY % (key)
                }
        
        if not (
            isinstance(filename, (str, list))
            and isinstance(model_path, str)
            and isinstance(predicted_filename, (str, list))
        ):
            return {
                "success": False,
                "message": ALL_STR_PARAM
            }
        
        # Checking if the mandatory parameters are all as expected.
        if not os.path.exists(model_path):
            str_ = model_path + " - Path does not exist."
            return return_response(str_)

        if is_list(filename) != is_list(predicted_filename):
            str_ = "filename and predicted_filename - both must be str or list"
            return return_response(str_)

        if is_string(filename):
            if not os.path.exists(filename):
                str_ = "filename: " + filename + " - Path does not exist."
                return return_response(str_)
            if not os.path.exists(predicted_filename):
                str_ = "pred: " + predicted_filename + " - Path does not exist."
                return return_response(str_)
        else:
            check = check_list_of_path_parameter(filename,
                                                EZ_IMAGE_SUPPORTED_TYPES,
                                                 len(predicted_filename),
                                                 "filename")
            if check is not None:
                return check
            check = check_list_of_path_parameter(predicted_filename,
                                            EZ_IMAGE_PRED_INPUT_FORMATS,
                                                 len(filename),
                                                 "predicted_filename")
            if check is not None:
                return check

        # If options is not empty, load the options, and check if they are as
        # expected.
        training_data = None
        score_strategy = None
        xai_strategy = None
        xai_image_path = None
        gradcam_layer = None
        model_num = None
        required_functions = None
        # print("options :", options)
        if 'training_data_path' in options:
            training_data = options['training_data_path']
            check = check_path_parameter(training_data,
                                         EZ_IMAGE_DATA_PATH_FORMAT,
                                         'training_data_path')
            if check is not None:
                return check

        if 'score_strategy' in options:
            score_strategy = options['score_strategy']
            check = check_radio_parameter(score_strategy,
                                          EZ_IMAGE_SCORE_STRATEGY,
                                          "score_strategy")
            if check is not None:
                return check

        if 'xai_strategy' in options:
            xai_strategy = options['xai_strategy']
            check = check_radio_parameter(xai_strategy,
                                          EZ_IMAGE_XAI_STRATEGY,
                                          "xai_strategy")
            if check is not None:
                return check

        if 'xai_image_path' in options:
            xai_image_path = options['xai_image_path']
            check = None
            if is_string(xai_image_path) != is_string(filename):
                str_ = "xai_image_path and filename must be list or str."
                check = return_response(str_)
            if check is not None:
                return check
            if not is_string(xai_image_path):
                if len(xai_image_path) != len(filename):
                    str_ = "xai_image_path diff length."
                    check = return_response(str_)
            if check is not None:
                return check

        if 'gradcam_layer' in options:
            gradcam_layer = options['gradcam_layer']
            if not is_string(gradcam_layer):
                str_ = "gradcam_layer: " + gradcam_layer
                str_ += " - must be a string."
                return return_response(str_)
        
        if 'model_num' in options:
            model_num = options['model_num']
            if not is_string(model_num):
                str_ = "model_num: " + model_num
                str_ += " - must be a string."
                return return_response(str_)

        if 'required_functions' in options:
            required_functions = options['required_functions']


        req = dict(
            image = filename,
            model = model_path,
            pred_image = predicted_filename,
            training_data = training_data,
            score_strategy = score_strategy,
            xai_strategy = xai_strategy,
            xai_image_path = xai_image_path,
            gradcam_layer = gradcam_layer,
            model_num = model_num,
            required_functions = required_functions,
            g = {}
        )

        results = exai_main_image.get_image_explainable_ai(req)

        return {
                "success": True,
                "message": EXPLANATION_SUCCESS,
                "explanations": results,
            }
    
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "message": INTERNAL_SERVER_ERROR
        }


def ez_image_active_learning(filenames,
                             model_path,
                             predicted_filenames,
                             options={}):
    """
    This API sorts test images based on explainability scores for the model's predictions.
    If a "query count" is specified in the options, it returns the indices and corresponding
    scores for that number of inputs.

    Parameters :
        - **filenames** (`list[str]`):
            List of absolute paths to the images for labeling (querying).
        - **model_path** (`str`) : 
            Path to the model. Must be an absolute path.\tSupported formats: `.h5`, `.pth`.
        - **predicted_filenames** (`list[str]`):
            List of absolute paths to the model's predictions. Must correspond to `filenames`. Supported formats: `.csv`, `.npy`.
        - **options** (`dict`):  
            Configuration for obtaining query information.
        
        .. code-block:: python

            options = {
                "query_count": 10,
                "training_data_path": "path/to/training/data.csv",
                "score_strategy": "weighted-moments",
                "al_strategy": "pool-based",
                "xai_strategy": "gradcam",
                "gradcam_layer": "layer_name",
                "model_num": "1"
            }

        **Key Fields**:
            - `query_count`: Number of test images to select for labeling.
            - `training_data_path`: Path to training data (absolute path; `.csv` format; columns: `inputs`, `labels`).
            - `score_strategy` (default: `"weighted-moments"`): Strategy for explainability scores.
            - `al_strategy`: Active Learning strategy (e.g., `"pool-based"`).
            - `xai_strategy` (default: `"gradcam"`): Strategy for image explanation. Required for `trainable-{1|2}` in `score_strategy`.
            - `gradcam_layer` (default: `layers[-2].name`): Layer name for gradcam if `score_strategy` is `trainable-{1|2}`.
            - `model_num` (default: `"1"`): Model ID if `score_strategy` is `trainable-*`.

    Returns :

        - **Dictionary with Fields**:
            - `success` (`bool`): Indicates success (`True`) or failure (`False`) of the API call.
            - `query_info` (`dict`): Dictionary containing:
                - `query_indices` (`list`): List of indices of test images selected for labeling.
                - `query_scores` (`list`): Corresponding explainability scores for the selected inputs.

        .. code-block:: python

            {
                "success": <True|False>,
                "query_info": {
                    "query_indices": [...],
                    "query_scores": [...]
                }
            }
    
    Example:
        .. code-block:: python

            ez_image_active_learning(
                filenames='classification',
                model_path='target',
                predicted_filenames='train.csv',
                options={"data_source": "parquet", "record_number": [1, 2, 3]}
            )
    """
    try:

        options = ez_modify_options_seg_model(options)
        
        for key in options:
            if key not in EZ_IMAGE_ACTIVE_OPTIONS_KEYS_LIST:
                return {
                    "success": False,
                    "message": INVALID_KEY % (key)
                }
            
        if (
            not is_string(model_path)
            or not is_list(filenames)
            or not is_list(predicted_filenames)
        ):
            return {
                "success": False,
                "message": ALL_STR_PARAM
            }

        if len(filenames) != len(predicted_filenames):
            str_ = "Filenames & predicted_filenames should have same length."
            return return_response(str_)

        if not os.path.exists(model_path):
            str_ = model_path + " - Path does not exist."
            return return_response(str_)

        check = check_list_of_path_parameter(predicted_filenames,
                                             EZ_IMAGE_PRED_INPUT_FORMATS,
                                             len(filenames),
                                             "predicted_filenames")
        if check is not None:
            return check
        check = check_list_of_path_parameter(filenames,
                                             EZ_IMAGE_SUPPORTED_TYPES,
                                             len(filenames),
                                             "filenames")
        if check is not None:
            return check

        query_count = None
        training_data = None
        score_strategy = None
        al_strategy = None
        xai_strategy = None
        gradcam_layer = None
        model_num = None
        required_functions = None

        if 'query_count' in options:
            query_count = options['query_count']
            check = check_if_positive_integer(query_count, "query_count")
            if check is not None:
                return check

        if 'training_data_path' in options:
            training_data = options['training_data_path']
            check = check_path_parameter(training_data,
                                         EZ_IMAGE_DATA_PATH_FORMAT,
                                         "training_data_path")
            if check is not None:
                return check

        if 'score_strategy' in options:
            score_strategy = options['score_strategy']
            check = check_radio_parameter(score_strategy,
                                          EZ_IMAGE_SCORE_STRATEGY,
                                          "score_strategy")
            if check is not None:
                return check

        if 'al_strategy' in options:
            al_strategy = options['al_strategy']
            check = check_radio_parameter(al_strategy,
                                          EZ_IMAGE_AL_STRATEGY,
                                          "al_strategy")
            if check is not None:
                return check

        if 'xai_strategy' in options:
            xai_strategy = options['xai_strategy']
            check = check_radio_parameter(xai_strategy,
                                          EZ_IMAGE_XAI_STRATEGY,
                                          "xai_strategy")
            if check is not None:
                return check
            
        if 'gradcam_layer' in options:
            gradcam_layer = options['gradcam_layer']
            if not is_string(gradcam_layer):
                str_ = "gradcam_layer: " + predicted_filenames
                str_ += " - must be a string."
                return return_response(str_)
                    
        if 'model_num' in options:
            model_num = options['model_num']
            if not is_string(model_num):
                str_ = "model_num: " + model_num
                str_ += " - must be a string."
                return return_response(str_)
        
        if 'required_functions' in options:
            required_functions = options['required_functions']
            
        req = dict(
            filenames = filenames,
            model_path = model_path,
            predicted_filenames = predicted_filenames,
            query_count = query_count,
            training_data = training_data,
            score_strategy = score_strategy,
            al_strategy = al_strategy,
            xai_strategy = xai_strategy,
            gradcam_layer = gradcam_layer,
            model_num = model_num,
            required_functions = required_functions,
            g = {}
        )

        results = exai_main_image.ez_active_learn_model(req)

        return {
                "success": True,
                "message": EXPLANATION_SUCCESS,
                "explanations": results,
            }
    
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "message": INTERNAL_SERVER_ERROR
        }


def ez_image_online_learning(new_training_data_path,
                             model_path,
                             options=None):
    """
    This API updates a given model using new training data and saves the updated model.
    The update process adapts based on the Online Learning strategy or optimizes performance on provided validation data.

    Parameters :
        - **new_training_data_path** (`str`):
            Path to the new data for training. Must be an absolute path. Supported format: `.csv`. Expected columns: `"inputs"`, `"labels"`.
        - **model_path** (`str`) :
            Path to the model. Must be an absolute path. Supported formats: `.h5`, `.pth`.

        - **options** (`dict`) :
            Configuration for training the model. Example:
    
        .. code-block:: python

            options = {
                "training_parameters": {...},
                "required_functions": {
                    "input_preprocess_fn": "pre-processing function",
                    "label_preprocess_fn: "label pre-processing function",
                    "output_process_fn": "Function to process outputs before applying the loss function.",
                    "loss_fn": "tensorflow loss function or custom loss function",
                    "metric_fns": {
                        "metric_iou" : "first metric function",
                        "metric_fscore" : "second metric function"
                    }
                },
                "training_data_path": "...",
                "validation_data_path": "...",
                "new_model_path": "...",
                "ol_strategy": "fine-tuning",
                "tr_strategy": "normal",
                "compiled": True,
                "log_file": "..."
            }

        **Key Fields**:
            - `"training_parameters"`: Dictionary with training settings, e.g., `"learning_rate"`, `"loss"`, `"metric"`, `"epochs"`, etc.
            - `"training_data_path"`: Path to previously used training data (required for `"joint-training"` strategy).
            - `"validation_data_path"`: Path to validation data for performance optimization.
            - `"new_model_path"`: Path to save the updated model.
            - `"ol_strategy"`: Online learning strategy (`"joint-training"`, `"fine-tuning"`, or `"knowledge-distillation"`; default: `"fine-tuning"`).
            - `"tr_strategy"`: Training strategy (`"normal"`, `"transfer-learning"`, `"dropouts"`; default: `"whole-model"`).

        **Required Functions**:
            - `"loss_fn"`: Loss function to train the model.
            - `"metric_fns"`: Dictionary of metric functions for evaluation.
            - `"input_preprocess_fn"`: Function to preprocess inputs.
            - `"label_preprocess_fn"`: Function to preprocess labels.
            - `"output_process_fn"`: Function to process outputs before applying the loss function.
    Returns :
        - **Dictionary with Fields** :
            - `success` (`bool`): Indicates success (`True`) or failure (`False`) of the API call.
            - `training_info` (`dict`): Contains the updated model path and training history, including losses and metrics for TensorFlow or PyTorch models.
    
    .. code-block:: python

        {
            "success": <True|False>,
            "training_info": {
                "model": "updated_model_path",
                "history": {
                    # Training history details
                }
            }
        }

    Example:
        .. code-block:: python

            ez_image_online_learning(
                training_data_path='classification',
                model_path='target',
                options={
                    "training_parameters": {
                        "batchsize": 4,
                        "epochs": 2,
                        "learning_rate": 1e-4
                        },
                    "ol_strategy": "fine-tuning",
                    "tr_strategy": "normal",
                    "validation_data_path": "validation/data/path.csv",
                    "new_model_path": "new/model/path.h5",
                    "required_functions": {
                        "input_preprocess_fn": "pre-processing function",
                        "label_preprocess_fn: "label pre-processing function",
                        "output_process_fn": "Function to process outputs before applying the loss function.",
                        "loss_fn": "tensorflow loss function or custom loss function",
                        "metric_fns": {
                            "metric_iou" : "first metric function",
                            "metric_fscore" : "second metric function"
                            }
                        },
                    "log_file": "path/to/log_file.csv"
            )
    """
    try:
        
        options = ez_modify_options_seg_model(options)
        model = model_path

        for key in options:
            if key not in EZ_IMAGE_ONLINE_OPTIONS_KEYS_LIST:
                return {
                    "success": False,
                    "message": INVALID_KEY % (key)
                }
        if (
            not is_string(model_path)
            or not is_string(new_training_data_path)
        ):
            return {
                "success": False,
                "message": ALL_STR_PARAM
            }

        check = check_path_parameter(new_training_data_path, 
                                    EZ_IMAGE_DATA_PATH_FORMAT,
                                    "new_training_data_path")
        if check is not None:
            return check

        check = check_path_parameter(model_path, 
                                    EZ_IMAGE_SUPPORTED_MODEL_FORMATS, 
                                    "model_path")
        if check is not None:
            return check

        training_data = None
        strategy = None
        validation_data = None
        new_model_path = None
        training_parameters = None
        tr_strategy = None
        required_functions = None
        compiled = None
        log_file = None
        print("options :", options)
        if 'training_data_path' in options:
            training_data = options['training_data_path']
            check = check_path_parameter(training_data,
                                         EZ_IMAGE_DATA_PATH_FORMAT,
                                         "training_data_path")
            if check is not None:
                return check

        if 'ol_strategy' in options:
            strategy = options['ol_strategy']
            check = check_radio_parameter(strategy,
                                          EZ_IMAGE_OL_STRATEGY,
                                          "ol_strategy")
            if check is not None:
                return check
            if strategy in ["joint-training", "knowledge-distillation"]:
                if 'training_data_path' not in options:
                    str_ = "training-data must be provided for JT and KD"
                    return return_response(str_, 422)

        if 'training_parameters' in options:
            training_parameters = options['training_parameters']
            if not isinstance(training_parameters, dict):
                str_ = "training_parameters should be a dict"
                return return_response(str_, 422)
            str_ = ""
            if "batch_size" in training_parameters and not\
                    isinstance(training_parameters['batch_size'], int):
                str_ = "batch_size should be integer"
                return return_response(str_, 422)
            elif "epochs" in training_parameters and not\
                    isinstance(training_parameters['epochs'], int):
                str_ = "epochs should be integer"
                return return_response(str_, 422)
            elif "learning_rate" in training_parameters and not\
                    isinstance(training_parameters['learning_rate'], 
                                float):
                str_ = "learning_rate should be float"
                return return_response(str_, 422)
            elif "dropout" in training_parameters and not\
                    isinstance(training_parameters['dropout'], float):
                str_ = "dropout should be float"
                return return_response(str_, 422)

        if 'tr_strategy' in options:
            tr_strategy = options['tr_strategy']
            check = check_radio_parameter(tr_strategy,
                                          EZ_IMAGE_TR_STRATEGY,
                                          "tr_strategy")
            if check is not None:
                return check
            if tr_strategy == "dropouts":
                str_ = "dropout must be provided for dropouts tr_strategy"
                if (training_parameters is None):
                    return return_response(str_, 422)
                if (training_parameters is not None):
                    if "dropout" not in training_parameters:
                        return return_response(str_, 422)
                    p = training_parameters["dropout"]
                    str_ = str(p) + " invalid dropout value"
                    if p >= 1 or p <= 0:
                        return return_response(str_, 422)

            if tr_strategy == "transfer-learning":
                str_ = "trainable_layers must be provided for transfer-learning"
                if (training_parameters is None):
                    return return_response(str_, 422)
                if (training_parameters is not None):
                    if "trainable_layers" not in training_parameters:
                        return return_response(str_, 422)
                    if check_if_positive_integer(
                        training_parameters["trainable_layers"],
                        "trainable_layers"):
                        return return_response(str_, 422)

        if 'validation_data_path' in options:
            validation_data = options['validation_data_path']
            check = check_path_parameter(validation_data,
                                         EZ_IMAGE_DATA_PATH_FORMAT,
                                         'validation_data_path')
            if check is not None:
                return check

        if 'new_model_path' in options:
            new_model_path = options['new_model_path']

        if 'required_functions' in options:
            required_functions = options['required_functions']
        
        if 'compiled' in options:
            compiled = options["compiled"]

        if 'log_file' in options:
            log_file = options["log_file"]

        req = dict(
            new_data = new_training_data_path,
            model = model,
            training_parameters = training_parameters,
            training_data = training_data,
            validation_data = validation_data,
            new_model_path = new_model_path,
            required_functions = required_functions,
            strategy = strategy,
            tr_strategy = tr_strategy,
            compiled = compiled,
            log_file = log_file,
            extra_info={}
        )

        response = exai_main_image.ez_online_learn(req)
        
        return response

    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "message": INTERNAL_SERVER_ERROR
        }
    

def ez_image_model_evaluate(validation_data_path,
                             model_path,
                             options=None):
    """
    This API validates a model using provided data and returns the model evaluation.

    Parameters :
        - **validation_data_path** (`str`):
            Path to the new data for validation. Must be an absolute path. Supported format: `.csv`. Expected columns: `"inputs"`, `"labels"`.
        - **model_path** (`str`):
            Path to the model. Supported formats: `.h5`, `.pth`. Must be an absolute path.
        - **options** (`dict`):
            Configuration for model training. Example:

        .. code-block:: python

            options = {
                "required_functions": {
                                    "loss_fn": '...',
                                    "metric_fns": '...',
                                    "input_preprocess_fn": '',
                                    "label_preprocess_fn": '',
                                    "output_process_fn": ''
                                    },
                "batch_size": 32,
                "log_file": "path/to/log/file"
            }

        - **Required Functions**:
            - `"loss_fn"`: Loss function to train the model.
            - `"metric_fns"`: Dictionary of metric functions for evaluation.
            - `"input_preprocess_fn"`: Function to preprocess inputs.
            - `"label_preprocess_fn"`: Function to preprocess labels.
            - `"output_process_fn"`: Function to process outputs before applying the loss function.

    Returns :
        - **Dictionary with Fields**:
            - `success` (`bool`): `True` if the call is successful, else `False`.
            - `eval_info` (`str`): Contains model evaluation details.
    
    Example:
        .. code-block:: python

            ez_image_model_evaluate(
                validation_data_path='path/to/image/image',
                model_path='path/to/model.h5',
                options = {
                        "batch_size": 4,
                        "required_functions": {
                            "input_preprocess_fn": "pre-processing function",
                            "label_preprocess_fn: "label pre-processing function",
                            "output_process_fn": "Function to process outputs before applying the loss function.",
                            "loss_fn": "tensorflow loss function or custom loss function",
                            "metric_fns": {
                                "metric_iou" : "first metric function",
                                "metric_fscore" : "second metric function"
                                }
                            }
                        }
            )
    """
    try:
        
        options = ez_modify_options_seg_model(options)

        #Check for valid keys in the options dict
        for key in options:
            if key not in EZ_IMAGE_EVALUATE_OPTIONS_KEYS_LIST:
                return {
                    "success": False,
                    "message": INVALID_KEY % (key)
                }
            
        val_data = validation_data_path
        model = model_path
        if (
            not is_string(model_path)
            or not is_string(val_data)
        ):
            return {
                "success": False,
                "message": ALL_STR_PARAM
            }

        check = check_path_parameter(val_data, 
                                    EZ_IMAGE_DATA_PATH_FORMAT,
                                    "validation_data_path")
        if check is not None:
            return check

        check = check_path_parameter(model_path, 
                                    EZ_IMAGE_SUPPORTED_MODEL_FORMATS, 
                                    "model_path")
        if check is not None:
            return check

        required_functions = None
        log_file = None
        batch_size = None
        print("options :", options)
        if 'required_functions' in options:
            required_functions = options['required_functions']

        if 'log_file' in options:
            log_file = options["log_file"]
            
        if "batch_size" in options and not isinstance(options['batch_size'], int):
            str_ = "batch_size should be integer"
            return return_response(str_, 422)

        req = dict(
            val_data = val_data,
            model = model_path,
            required_functions = required_functions,
            batch_size = batch_size,
            log_file = log_file,
            extra_info={}
        )

        response = exai_main_image.ez_model_evaluate(req)

        return response
    
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "message": INTERNAL_SERVER_ERROR
        }
