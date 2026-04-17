from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel, field_validator
from predict import PredictionRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from jinja2 import Environment, FileSystemLoader
from fastapi.responses import HTMLResponse
 
import torch
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(BASE_DIR, "templates")
static_dir = os.path.join(BASE_DIR, "static")

# Create directories if they don't exist
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

# Initialize templates
templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")



# Initialize app state
app.state.model = None
app.state.X_test = None
app.state.Y_test = None
app.state.optimal_threshold = 0.5
#app.state.normalization_transformer = None

# Import the model at the start of the application if it exists
model_path = os.path.join(os.path.dirname(__file__), '../models/model.joblib')
normalization_path = os.path.join(os.path.dirname(__file__), '../pipelines/pipeline_config.pkl')
threshold_path = os.path.join(os.path.dirname(__file__), '../thresholds/threshold_model.joblib')

# # Verify existing imported files and load them
# # if os.path.exists(model_path):
# #     app.state.model = joblib.load(model_path)
# #     print(f"Modèle chargé depuis: {model_path}")
# # else:
# #     app.state.model = None
# #     print(f"Modèle non trouvé à {model_path}. Veuillez appeler /train pour entraîner un modèle.")
if os.path.exists(model_path):
    app.state.model = joblib.load(model_path)
    print(f"Modèle chargé depuis: {model_path}")
else:
    app.state.model = None
    print(f"Modèle non trouvé à {model_path}. Veuillez appeler /train pour entraîner un modèle.")

# if os.path.exists(model_path):
#     try:
#         app.state.model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
#         if hasattr(app.state.model, 'eval'):
#             app.state.model.eval()
#         print(f"Model loaded from: {model_path}")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         app.state.model = None
# else:
#     print(f"Model not found at {model_path}")

# Load normalization
if os.path.exists(normalization_path):
    config = joblib.load(normalization_path)
    # Extract the normalization transformer from the config dict
    app.state.normalization_transformer = config.get("normalization_transformer", None)
    print(f"Normalisation transformer chargé depuis: {normalization_path}")
else:
    app.state.normalization_transformer = None
    print(f"Normalisation transformer non trouvé à {normalization_path}. Assurez-vous que le pipeline de normalisation est sauvegardé.")
# if os.path.exists(normalization_path):
#     try:
#         config = joblib.load(normalization_path)
#         if isinstance(config, dict):
#             app.state.normalization_transformer = config.get("normalization_transformer") or config.get("transformer")
#         else:
#             app.state.normalization_transformer = config
#         if app.state.normalization_transformer:
#             print(f"Normalization loaded")
#         else:
#             print(f"No transformer found")
#     except Exception as e:
#         print(f"Could not load normalization: {e}")
# else:
#     print(f"Normalization file not found")



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
# Setup Jinja2 environment
env = Environment(
    loader=FileSystemLoader(templates_dir),
    cache_size=0, 
    auto_reload=True
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    template = env.get_template("form.html")
    html_content = template.render(request=request)
    return HTMLResponse(content=html_content)
# @app.get("/")
# def read_root():
#     return {"message": "Welcome in our deep Learning project!"}


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
# @app.post("/predict")
# def predict(input_data: PredictionRequest):
#     try:
#         data_dict = input_data.model_dump()
#         data_dict.pop('Target', None)
        
#         X_raw = pd.DataFrame([data_dict])
#         print(f"Input: {X_raw.iloc[0].to_dict()}")
        
#         # Apply normalization if available
#         if app.state.normalization_transformer is not None:
#             try:
#                 X_processed = app.state.normalization_transformer.transform(X_raw)
#                 print("Normalization applied")
#             except Exception as e:
#                 print(f"Normalization failed: {e}")
#                 X_processed = X_raw
#         else:
#             X_processed = X_raw
        
#         # Make prediction
#         if app.state.model is not None:
#             y_pred, y_pred_proba = make_prediction(app.state.model, X_processed, threshold=app.state.optimal_threshold)
#         else:
#             # Mock prediction
#             risk_score = (data_dict.get('BMI', 25) / 50) + (data_dict.get('Age', 5) / 26)
#             y_pred_proba = [min(0.95, max(0.05, risk_score / 3))]
#             y_pred = [1 if y_pred_proba[0] > app.state.optimal_threshold else 0]
#             print("Using mock prediction")
        
#         return {
#             "prediction": int(y_pred[0]),
#             "probability": float(y_pred_proba[0]),
#             "threshold": app.state.optimal_threshold
#         }
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)






