import os
import argparse
import pandas as pd
import joblib
from pathlib import Path
from .processing_class import (
    get_processing_pipeline,
    DiabetesTargetBinarizer,
    StratifiedSplitter,
    SEED
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

from sklearn import set_config
set_config(transform_output="pandas")

def main():
    parser = argparse.ArgumentParser(description="Generate data for the project.")
    parser.add_argument("--input", type=str, default=os.getenv("DATA_INPUT_PATH", "dataset/diabetes_012_health_indicators_BRFSS2015.csv"), help="Path to input data")
    parser.add_argument("--output", type=str, default=os.getenv("DATA_OUTPUT_DIR", "data"), help="Directory to save generated data")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    print(f"Generating data from {input_path} to {output_dir}...")
    
    # Check if input exists
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    raw_df = pd.read_csv(input_path)
    
    # Define features logic
    columns_to_drop = ["AnyHealthcare", "NoDocbcCost", "Fruits", "Veggies", "Sex", "Smoker"]
    binary_vars = [col for col in raw_df.columns if col not in columns_to_drop and raw_df[col].nunique() == 2]
    # Note: 'target' is added after binarization, so we assume its existence for the transformer mapping
    non_transformed_features = binary_vars + ["Education", "BMI_category", "target"]

    # Get pipelines
    data_cleaning_pipeline, normalization_transformer = get_processing_pipeline(
        columns_to_drop=columns_to_drop,
        binary_vars=binary_vars,
        non_transformed_features=non_transformed_features
    )

    # 1. Binarize Target
    print("Binarizing target...")
    binarizer = DiabetesTargetBinarizer(column_name="Diabetes_012", output_column_name="target", drop_original=True)
    df_binarized = binarizer.transform(raw_df)

    # 2. Clean Data
    print("Cleaning data...")
    df_cleaned = data_cleaning_pipeline.fit_transform(df_binarized)

    # 3. Split Data
    print("Splitting data...")
    splitter = StratifiedSplitter(test_size=0.3, val_size=0.5, random_state=SEED, y_column_name="target")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splitter.transform(df_cleaned)

    # 4. Normalize
    print("Normalizing data...")
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    normalization_transformer.fit(train_df)
    
    train_norm = normalization_transformer.transform(train_df)
    val_norm = normalization_transformer.transform(val_df)
    test_norm = normalization_transformer.transform(test_df)

    # 5. Resample training data
    print("Resampling training data...")
    oversampler = SMOTE(random_state=SEED, sampling_strategy=0.6)
    undersampler = NearMiss(version=1, sampling_strategy=0.7)
    
    X_train_resampled, y_train_resampled = oversampler.fit_resample(
        train_norm.drop("target", axis=1), train_norm["target"]
    )
    X_train_resampled, y_train_resampled = undersampler.fit_resample(
        X_train_resampled, y_train_resampled
    )
    train_norm = pd.concat([X_train_resampled, y_train_resampled], axis=1)

    # Save
    print(f"Saving files to {output_dir}...")
    train_norm.to_csv(output_dir / "train_data.csv", index=False)
    val_norm.to_csv(output_dir / "validation_data.csv", index=False)
    test_norm.to_csv(output_dir / "test_data.csv", index=False)

    joblib.dump(data_cleaning_pipeline.named_steps["categorize_bmi"], output_dir / "categorize_bmi.joblib")
    joblib.dump(normalization_transformer, output_dir / "normalization_transformer.joblib")
    print(f"Data processing pipelines saved to {output_dir}")
    print(f"Training set: {train_norm.shape[0]} samples, {train_norm.shape[1]} features (after resampling)")
    print(f"Validation set: {val_norm.shape[0]} samples, {val_norm.shape[1]} features")
    print(f"Test set: {test_norm.shape[0]} samples, {test_norm.shape[1]} features")
    print("Data generation complete.")

if __name__ == "__main__":
    main()
