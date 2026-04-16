from fastapi import FastAPI, HTTPException
from .predict import PredictionRequest
from .metrics import calculate_metrics, make_prediction
from .cache_loading import recreate_preprocessing_pipeline, load_model_with_cached_threshold
from .cli_parameters import parse_cli_parameters
import pandas as pd
import numpy as np

app = FastAPI(title="Deep Learning Project")

# Initialize app state
app.state.model = None
app.state.X_test = None
app.state.Y_test = None
app.state.optimal_threshold = 0.5
app.state.processing_pipeline  = None

@app.on_event("startup")
async def app_init():
    """
    Initialize the FastAPI application by loading the model, normalization transformer, and optimal threshold from cache.
    
    This function is called at the start of the application to ensure that all necessary components are loaded and ready for predictions and metrics calculations.
    
    It checks for the existence of the model, normalization transformer, and threshold files in the specified cache directories. If they exist, it loads them into the application state. If not, it logs appropriate messages and sets default values where necessary.
    
    The cache paths are configured via CLI parameters (--model-cache and --data-cache) or environment variables (MODEL_CACHE and DATA_CACHE).
    
    Returns:
        None
    """
    print("Starting application initialization...")
    # Pass empty list to parse_cli_parameters to skip uvicorn's sys.argv and only use environment variables
    parameters = parse_cli_parameters(args=[])
    model_cache_path = parameters.model_cache
    data_cache_path = parameters.data_cache

    # Load model, normalization transformer, and optimal threshold
    app.state.model, app.state.optimal_threshold = load_model_with_cached_threshold(model_cache_path)
    app.state.processing_pipeline = recreate_preprocessing_pipeline(data_cache_path)
    
    print("Application initialized with cached model, normalization transformer, and optimal threshold.")

# Define API endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome in our deep Learning project!"}


@app.post("/metrics")
def get_metrics():
    """
    Calculate metrics on test data (with labels).
    Returns: accuracy, precision, recall, f1, confusion_matrix
    """
    if app.state.model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Ensure model.joblib exists.")
    
    if app.state.X_test is None or app.state.Y_test is None:
        raise HTTPException(status_code=400, detail="Test data not available. Ensure test_data.csv exists.")
    
    X_test = app.state.X_test
    Y_test = app.state.Y_test
    
    # Calculate metrics on test data with optimal threshold
    metrics_result = calculate_metrics(app.state.model, X_test, Y_test, threshold=0.5)
    return metrics_result

@app.post("/predict")
def predict(input_data: PredictionRequest):
    """
    Make prediction on new raw data
    """
    try:
        # Convert to dict and remove Target (it's not a feature, only for training)
        data_dict = input_data.model_dump()
        # data_dict.pop('Target', None)  # Remove Target field
        data_dict["target"] = 0 # Add target to satisfy pipeline, but it is ignore for transformation
        
        X_raw = pd.DataFrame([data_dict])
        print(f"Input data received for prediction: {X_raw}")
        print(f"Number of features: {X_raw.columns}")
        X_processed = app.state.processing_pipeline.transform(X_raw).drop("target", axis=1)  # Drop the target column after transformation
        print(f"Processed data for prediction: {X_processed}")
        print(f"Number of features after processing: {X_processed.columns}")
        
        # Make the prediction with normalized data
        y_pred, y_pred_proba = make_prediction(app.state.model, X_processed, threshold=app.state.optimal_threshold)

        return {
            "prediction": int(y_pred[0]),
            "probability": float(y_pred_proba[0]),
            "threshold": app.state.optimal_threshold
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)