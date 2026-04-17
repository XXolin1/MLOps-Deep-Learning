import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn import set_config
import torch

set_config(transform_output="pandas")


def recreate_preprocessing_pipeline(path_to_data=None):
    """
    Recreate the preprocessing pipeline that includes categorization, normalization, and feature dropping.
    
    This function loads or creates:
    1. The BMI Categorizer transformer (categorization step)
    2. The normalization transformer (normalization step for various features)
    3. A feature dropper transformer that drops the target column after transformation
    
    Returns:
        Pipeline: Complete preprocessing pipeline with all transformation steps
    
    Raises:
        FileNotFoundError: If required preprocessing files are not found
    """
    if path_to_data is None:
        raise FileNotFoundError(
            "Path to data directory is required to load preprocessing components. "
            "Please provide the path to the data directory containing the saved transformers."
        )

    path_to_data = Path(path_to_data)

    # Try to load the saved normalization transformer
    normalization_path = path_to_data / "normalization_transformer.joblib"
    
    if not normalization_path.exists():
        raise FileNotFoundError(
            f"Normalization transformer not found at {normalization_path}. "
            "Please run data generation first using generate_data.sh or generate_data.bat"
        )
    
    normalization_transformer = joblib.load(normalization_path)
    normalization_transformer.set_output(transform="pandas")  # Ensure output is a DataFrame for compatibility with the rest of the pipeline
    
    categorization_path = path_to_data / "categorize_bmi.joblib"

    if not categorization_path.exists():
        raise FileNotFoundError(
            f"Categorization pipeline not found at {categorization_path}. "
            "Please run data generation first using generate_data.sh or generate_data.bat"
        )

    categorization_pipeline = joblib.load(categorization_path)

    pipeline = Pipeline([
        ("categorization", categorization_pipeline),
        ("normalization", normalization_transformer),
    ])

    return pipeline


def load_model_with_cached_threshold(path_to_model=None):
    """
    Load the trained AI model and the cached optimal threshold for predictions.
    
    This function loads:
    1. The trained PyTorch/neural network model
    2. The cached optimal threshold used for binary classification
    
    Returns:
        tuple: (model, optimal_threshold)
               - model: Trained neural network model object
               - optimal_threshold: Cached optimal threshold (float) for predictions
    
    Raises:
        FileNotFoundError: If model or threshold files are not found
    """

    if path_to_model is None:
        raise FileNotFoundError(
            "Path to model directory is required to load the model and threshold. "
            "Please provide the path to the model directory containing the saved model and threshold."
        )

    path_to_model = Path(path_to_model)

    # Load the model - try CUDA first, then fall back to CPU
    model_path = path_to_model / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train a model first using train_model.sh or train_model.bat"
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device, weights_only=False)
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "device"):
        model.device = device
    
    # Load the cached optimal threshold
    threshold_path = path_to_model / "threshold.joblib"
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Optimal threshold not found at {threshold_path}. "
            "Please train a model first to calculate the optimal threshold"
        )
    
    optimal_threshold = joblib.load(threshold_path)
    
    return model, optimal_threshold