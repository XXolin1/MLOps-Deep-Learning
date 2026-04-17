from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from predict import PredictionRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import joblib
import os
import sys # test
import pandas as pd
import numpy as np

# Add deliverables path to sys.path to load custom classes from data_preparation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../deliverables')) # test

# function import
from metrics import calculate_metrics, make_prediction

app = FastAPI(title="Deep Learning Project")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize app state
app.state.model = None
app.state.X_test = None
app.state.Y_test = None
app.state.optimal_threshold = 0.5

# Import the model at the start of the application if it exists
model_path = os.path.join(os.path.dirname(__file__), '../models/model.joblib')
normalization_path = os.path.join(os.path.dirname(__file__), '../pipelines/pipeline_config.pkl')
threshold_path = os.path.join(os.path.dirname(__file__), '../thresholds/threshold_model.joblib')

# Verify existing imported files and load them
if os.path.exists(model_path):
    app.state.model = joblib.load(model_path)
    print(f"Modèle chargé depuis: {model_path}")
else:
    app.state.model = None
    print(f"Modèle non trouvé à {model_path}. Veuillez appeler /train pour entraîner un modèle.")

if os.path.exists(normalization_path):
    config = joblib.load(normalization_path)
    # Extract the normalization transformer from the config dict
    app.state.normalization_transformer = config.get("normalization_transformer", None)
    print(f"Normalisation transformer chargé depuis: {normalization_path}")
else:
    app.state.normalization_transformer = None
    print(f"Normalisation transformer non trouvé à {normalization_path}. Assurez-vous que le pipeline de normalisation est sauvegardé.")

#Load optimal threshold
if os.path.exists(threshold_path):
    app.state.optimal_threshold = joblib.load(threshold_path)
    print(f"Threshold chargé: {app.state.optimal_threshold}")
else:
    app.state.optimal_threshold = 0.5
    print(f"Threshold non trouvé, utilisation de 0.5 par défaut")

# Load test data if available
test_data_path = os.path.join(os.path.dirname(__file__), '../src/deliverables/test_data.csv')
if os.path.exists(test_data_path):
    try:
        test_dataset = pd.read_csv(test_data_path)
        app.state.Y_test = test_dataset["target"]
        app.state.X_test = test_dataset.drop(["target"], axis=1)
        print(f"Données de test chargées: {len(app.state.X_test)} samples")
    except Exception as e:
        print(f"Erreur lors du chargement des données de test: {e}")
else:
    print(f"Données de test non trouvées à {test_data_path}")


# Define API endpoints
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


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
        data_dict.pop('Target', None)  # Remove Target field
        
        X_raw = pd.DataFrame([data_dict])
        print(f"Input data received for prediction: {X_raw}")
        print(f"Number of features: {X_raw.shape[1]}")
        X_normalized = app.state.normalization_transformer.transform(X_raw)
        print(f"Data after normalization: {X_normalized}")
        
        # Faire la prédiction
        y_pred, y_pred_proba = make_prediction(app.state.model, X_raw, threshold=0.5)
        
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