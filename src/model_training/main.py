import os
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .model_class import PyTorchMLP
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import numpy as np
import copy
import joblib

def main():
    parser = argparse.ArgumentParser(description="Train model for the project.")
    parser.add_argument("--data-dir", type=str, default=os.getenv("DATA_DIR", "data"), help="Directory where data is stored")
    parser.add_argument("--model-out", type=str, default=os.getenv("MODEL_OUTPUT_DIR", "models"), help="Directory to save the trained model")
   
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_out)
    
    print(f"Training model with data from {data_dir}...")
    print(f"Model will be saved to {model_dir}")
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Loading data set and split into features and target
    train_dataset = pd.read_csv(data_dir / "train_data.csv")
    test_dataset = pd.read_csv(data_dir / "test_data.csv")
    val_dataset = pd.read_csv(data_dir / "validation_data.csv")

    X_train, y_train = train_dataset.drop("target", axis=1), train_dataset["target"]
    X_test, y_test = test_dataset.drop("target", axis=1), test_dataset["target"]
    X_val, y_val = val_dataset.drop("target", axis=1), val_dataset["target"]

    print("=== Setting up Cross Validation ===")

    # Create StratifiedKFold object
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Combine train and validation data for cross validation
    X_cv = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_cv = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    cv_accuracies = []
    cv_f1_scores = []

    best_cv_model = None
    best_cv_f1 = -1
    best_fold = -1

    fold = 1

    for train_idx, val_idx in skf.split(X_cv, y_cv):
        print(f"\n--- Training Fold {fold}/{n_splits} ---")
        
        # Split data for the current fold
        X_train_fold, X_val_fold = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
        y_train_fold, y_val_fold = y_cv.iloc[train_idx], y_cv.iloc[val_idx]
        
        # Initialize a fresh model
        cv_model = PyTorchMLP(
            input_size=len(X_cv.columns),
            hidden_layers=[128, 64, 16],
            output_size=1,
            activation="tanh",
            output_activation="sigmoid",
            loss_function='bce',
            optimizer='adam',
            learning_rate=0.001,
            epochs=100,
            batch_size=512,
            metrics='f1_score',
            dropout=0.5,
            # l2_reg=0.001,
            early_stopping=True,
            early_stopping_patience=35,
            early_stopping_min_delta=0,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Fit on fold data
        cv_model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
        
        # Predict and evaluate on validation fold
        y_pred_proba = cv_model.predict(X_val_fold)
        y_pred_fold = (y_pred_proba > 0.5).astype(int)
        
        acc = accuracy_score(y_val_fold, y_pred_fold)
        f1 = f1_score(y_val_fold, y_pred_fold)
        
        print(f"Fold {fold} Results - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
        
        cv_accuracies.append(acc)
        cv_f1_scores.append(f1)
        
        # Memorize the best model based on F1-Score
        if f1 > best_cv_f1:
            best_cv_f1 = f1
            best_cv_model = copy.deepcopy(cv_model)
            best_fold = fold
            
        fold += 1

    print("\n=== Cross Validation Results ===")
    print(f"Mean Accuracy: {np.mean(cv_accuracies):.4f} +/- {np.std(cv_accuracies):.4f}")
    print(f"Mean F1-Score: {np.mean(cv_f1_scores):.4f} +/- {np.std(cv_f1_scores):.4f}")
    print(f"Best Model found in Fold {best_fold} with F1-Score: {best_cv_f1:.4f}")
    print("Model training complete.")

    # ===== THRESHOLD ANALYSIS =====
    print("\n=== THRESHOLD ANALYSIS ===")

    # Get predictions with probabilities
    y_pred_proba = best_cv_model.predict(X_test).flatten()

    # Test different thresholds
    threshold_step = 0.01  # Configurable step for threshold exploration
    thresholds = np.arange(0.1, 1.0, threshold_step)
    results_thresholds = []

    for output_threshold in thresholds:
        y_pred = (y_pred_proba > output_threshold).astype(int)
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        
        results_thresholds.append({
            'threshold': output_threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    # Find best threshold by F1 score
    best_result = max(results_thresholds, key=lambda x: x['f1'])
    optimal_threshold = best_result['threshold']

    print(f"\n=== OPTIMAL THRESHOLD ===")
    print(f"Best threshold (by F1): {optimal_threshold:.2f}")
    print(f"Accuracy:  {best_result['accuracy']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall:    {best_result['recall']:.4f}")
    print(f"F1 Score:  {best_result['f1']:.4f}")

    torch.save(best_cv_model, model_dir / "model.pt")
    print(f"Best model saved to {model_dir / 'model.pt'}")
    joblib.dump(optimal_threshold, model_dir / "threshold.joblib")
    print(f"Optimal threshold saved to {model_dir / 'threshold.joblib'}")

if __name__ == "__main__":
    main()
