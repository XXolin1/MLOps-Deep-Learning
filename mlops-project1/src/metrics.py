from pyexpat import model

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for FastAPI
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import numpy as np
import os


# ===== SIMPLE PREDICTION =====
def make_prediction(model, X_data, threshold=0.5):
    """Make predictions on data"""
    y_pred_proba = model.predict(X_data).flatten()
    y_pred = (y_pred_proba > threshold).astype(int)
    return y_pred, y_pred_proba


# ===== CALCULATE METRICS =====
def calculate_metrics(model, X_test, Y_test, threshold=0.5):
    """Calculate metrics on test data"""
    y_pred, y_pred_proba = make_prediction(model, X_test, threshold)
    
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred, zero_division=0)
    recall = recall_score(Y_test, y_pred, zero_division=0)
    f1 = f1_score(Y_test, y_pred, zero_division=0)
    cm = confusion_matrix(Y_test, y_pred)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }