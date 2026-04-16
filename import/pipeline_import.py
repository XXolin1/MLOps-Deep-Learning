import joblib
import pandas as pd
from pathlib import Path

# Important: To load the pipeline, custom classes must be defined
import delivrables.data_preparation as dp

# Paths
DATA_DIR = Path("dataset")
OUTPUT_DIR = Path("deliverables")
DATA_PATH = DATA_DIR / "diabetes_012_health_indicators_BRFSS2015.csv"

# 1. Load the raw data
print("Loading raw data...")
raw_df = pd.read_csv(DATA_PATH)

# 2. Load the pipeline configuration
print("Loading pipeline configuration...")
config = joblib.load(OUTPUT_DIR / "pipeline_config.pkl")

# Extract the fitted transformers and splitters
# (These names match the dictionary keys used in data_preparation.py)
diabetes_binarizer = dp.DiabetesTargetBinarizer(
    column_name="Diabetes_012", 
    output_column_name="target", 
    drop_original=True
)

data_cleaning_pipeline = dp.Pipeline([
    ("feature_dropper", config["feature_dropper"]),
    ("missing_imputation", config["missing_imputation"]),
    ("categorize_bmi", config["categorize_bmi"]),
    ("deduplication", config["deduplication"]),
])

stratified_splitter = config["stratified_splitter"]
normalization_transformer = config["normalization_transformer"]

# 3. Apply the pipeline steps in order
# print("Binarizing target...")
# diabete_binary_df = diabetes_binarizer.transform(raw_df)

# print("Cleaning data...")
# clean_df = data_cleaning_pipeline.transform(diabete_binary_df)

# print("Splitting data...")
# (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_splitter.transform(clean_df)

# # Construct dataframes for normalization
# train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
# val_df = pd.merge(X_val, y_val, left_index=True, right_index=True)
# test_df = pd.merge(X_test, y_test, left_index=True, right_index=True)

# print("Normalizing data...")
# # Use the fitted normalization transformer from the config
# train_norm_df = normalization_transformer.transform(train_df)
# val_norm_df = normalization_transformer.transform(val_df)
# test_norm_df = normalization_transformer.transform(test_df)

# # 4. Final summary
# print("\n--- Processing Complete ---")
# print(f"Original shape:    {raw_df.shape}")
# print(f"Cleaned shape:     {clean_df.shape}")
# print(f"Train Norm shape:  {train_norm_df.shape}")
# print(f"Val Norm shape:    {val_norm_df.shape}")
# print(f"Test Norm shape:   {test_norm_df.shape}")