# #### Imports & Configurations

# %%
# Maths, data manipulation and visualization libraries
import numpy as np
import pandas as pd

# Scikit-learn libraries for data processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.utils.validation import check_is_fitted

from sklearn import set_config
set_config(transform_output="pandas")

# Exportaion
import joblib

# Misc libraries
import random
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
DATA_DIR = Path("dataset")
OUTPUT_DIR = Path("deliverables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "diabetes_012_health_indicators_BRFSS2015.csv"

# %% [markdown]
# #### Loading

# %%
raw_df = pd.read_csv(DATA_PATH)

# %% [markdown]
# ## 10. Pipeline Transformation algorithm (MLOps)

# %%
class BMI_Categorizer(BaseEstimator, TransformerMixin):
    def __init__(self, input_column="BMI", output_column="BMI_category", drop_original=True):
        self.input_column = input_column
        self.output_column = output_column
        self.drop_original = drop_original
    
    @staticmethod
    def categorise_bmi(bmi):
        if bmi < 18.5:
            return 0
        elif 18.5 <= bmi < 25:
            return 1
        elif 25 <= bmi < 30:
            return 2
        else:
            return 3

    def __sklearn_is_fitted__(self):
        # This method is required by scikit-learn to check if the transformer is fitted
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_copy = X.copy()
        X_copy[self.output_column] = X_copy[self.input_column].apply(self.categorise_bmi)
        if self.drop_original:
            X_copy = X_copy.drop(self.input_column, axis=1)
        return X_copy

class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def __sklearn_is_fitted__(self):
        # This method is required by scikit-learn to check if the transformer is fitted
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        check_is_fitted(self)
        return X.drop(columns=self.columns_to_drop)

# Square root transformer
class SqrtTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def __sklearn_is_fitted__(self):
        # This method is required by scikit-learn to check if the transformer is fitted
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_copy = X.copy()
        for col in X.columns:
            X_copy[col] = np.sqrt(X_copy[col])
        return X_copy

# Clamp transformer
class ClampTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, upper=None, lower=None):
        self.upper = upper
        self.lower = lower

    def __sklearn_is_fitted__(self):
        # This method is required by scikit-learn to check if the transformer is fitted
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_copy = X.copy()
        for col in X.columns:
            if self.upper is not None:
                X_copy[col] = np.minimum(X_copy[col], self.upper)
            if self.lower is not None:
                X_copy[col] = np.maximum(X_copy[col], self.lower)
        return X_copy

class ReflectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def __sklearn_is_fitted__(self):
        # This method is required by scikit-learn to check if the transformer is fitted
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_copy = X.copy()
        for col in X.columns:
            X_copy[col] = np.max(X_copy[col] + 1) - X_copy[col]
        return X_copy

class DeduplicationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column="target"):
        self.target_column = target_column

    def __sklearn_is_fitted__(self):
        # This method is required by scikit-learn to check if the transformer is fitted
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_clean = X.drop_duplicates().copy()
        
        # Remove confused rows: duplicates where features are identical but target differs
        # except when the target is 1.
        features_cols = [c for c in X_clean.columns if c != self.target_column]
        is_confused = X_clean.duplicated(subset=features_cols, keep=False)
        
        # Keep non-confused rows OR rows with target == 1
        X_final = X_clean[ (~is_confused) | (X_clean[self.target_column] == 1) ].copy()
        
        # Ensure no exact feature-level duplicates remain for the kept target=1 rows
        return X_final.drop_duplicates(subset=features_cols)

class MissingValueImputationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="drop"):
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.strategy == "median":
            self.fill_values_ = X.median()
        elif self.strategy == "mean":
            self.fill_values_ = X.mean()
        elif self.strategy == "most_frequent":
            self.fill_values_ = X.mode().iloc[0]
        elif self.strategy == "drop":
            self.fill_values_ = None
        else:
            raise ValueError("Invalid strategy. Use 'median', 'mean', or 'most_frequent'.")
        return self

    def transform(self, X):
        check_is_fitted(self)
        if self.fill_values_ is None:
            return X.dropna()
        return X.fillna(self.fill_values_)

class DiabetesTargetBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name="Diabetes_012", output_column_name="target", drop_original=True):
        self.column_name = column_name
        self.output_column_name = output_column_name
        self.drop_original = drop_original

    def __sklearn_is_fitted__(self):
        # This method is required by scikit-learn to check if the transformer is fitted
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_copy = X.copy()
        X_copy[self.output_column_name] = X_copy[self.column_name].apply(lambda x: 1 if x == 2 else 0)
        if self.drop_original:
            X_copy = X_copy.drop(self.column_name, axis=1)
        return X_copy

class StratifiedSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, test_size=0.3, val_size=0.5, y_column_name=None, random_state=None):
        self.test_size = test_size
        self.val_size = val_size
        self.y_column_name = y_column_name
        self.random_state = random_state

    def __sklearn_is_fitted__(self):
        # This method is required by scikit-learn to check if the transformer is fitted
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        check_is_fitted(self)
        y = X[self.y_column_name] if self.y_column_name else None
        X = X.drop(columns=[self.y_column_name]) if self.y_column_name else X
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size,
            stratify=y_temp,
            random_state=self.random_state
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

clamp_minmax_transformer = Pipeline([
    ("clamp", ClampTransformer(upper=15)),
    ("minmax", MinMaxScaler())
])

reflect_sqrt_transformer = Pipeline([
    ("reflect", ReflectTransformer()),
    ("sqrt", SqrtTransformer())
])

columns_to_drop = ["AnyHealthcare", "NoDocbcCost", "Fruits", "Veggies", "Sex", "Smoker"]
binary_vars = [col for col in raw_df.columns if col not in columns_to_drop and raw_df[col].nunique() == 2]
non_transformed_features = binary_vars + ["Education", "BMI_category"] + ["target"]

feature_dropper = FeatureDropper(columns_to_drop=columns_to_drop)

normalization_transformer = ColumnTransformer([
    ("minmax", MinMaxScaler(), ["Income"]),
    ("clamp", clamp_minmax_transformer, ["MentHlth", "PhysHlth"]),
    ("reflect_sqrt", reflect_sqrt_transformer, ["Age"]),
    ("std", StandardScaler(), ["GenHlth"]),
    ("none", "passthrough", non_transformed_features)
], verbose_feature_names_out=False)

diabetes_binarizer = DiabetesTargetBinarizer(column_name="Diabetes_012", output_column_name="target", drop_original=True)

stratified_splitter = StratifiedSplitter(test_size=0.3, val_size=0.5, random_state=SEED, y_column_name="target")

oversampler = SMOTE(random_state=SEED, sampling_strategy=0.6)
undersampler = NearMiss(version=1, sampling_strategy=0.7)

data_cleaning_pipeline = Pipeline([
    ("feature_dropper", feature_dropper),
    ("missing_imputation", MissingValueImputationTransformer(strategy="drop")),
    ("categorize_bmi", BMI_Categorizer(input_column="BMI", output_column="BMI_category", drop_original=True)),
    ("deduplication", DeduplicationTransformer()),
])


###### run sur la prepa

# %%
raw_df = pd.read_csv(DATA_PATH)
diabete_binary_df = diabetes_binarizer.fit_transform(raw_df)
clean_df = data_cleaning_pipeline.fit_transform(diabete_binary_df)
(X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_splitter.fit_transform(clean_df)

train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
val_df = pd.merge(X_val, y_val, left_index=True, right_index=True)
test_df = pd.merge(X_test, y_test, left_index=True, right_index=True)

train_df = normalization_transformer.fit_transform(train_df)
X_train, y_train = oversampler.fit_resample(train_df.drop("target", axis=1), train_df["target"])
X_train, y_train = undersampler.fit_resample(X_train, y_train)
train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)

val_df = normalization_transformer.transform(val_df)

test_df = normalization_transformer.transform(test_df)

# %% [markdown]
# ## 11. Saving of the cleaned dataset

# %% [markdown]
# To complete this document, we will now save the cleaned dataset into multiple CSV files

# %%
# Save to CSV files
train_path = OUTPUT_DIR / "train_data.csv"
val_path = OUTPUT_DIR / "validation_data.csv"
test_path = OUTPUT_DIR / "test_data.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Training data saved to: {train_path}")
print(f"Validation data saved to: {val_path}")
print(f"Test data saved to: {test_path}")
print()
print(f"Training set: {train_df.shape[0]} samples, {train_df.shape[1]} features")
print(f"Validation set: {val_df.shape[0]} samples, {val_df.shape[1]} features")
print(f"Test set: {test_df.shape[0]} samples, {test_df.shape[1]} features")

# Save pipeline configuration for reproducibility
pipeline_config = {
    "feature_dropper": feature_dropper,
    "missing_imputation": data_cleaning_pipeline.named_steps["missing_imputation"],
    "categorize_bmi": data_cleaning_pipeline.named_steps["categorize_bmi"],
    "deduplication": data_cleaning_pipeline.named_steps["deduplication"],
    "stratified_splitter": stratified_splitter,
    "normalization_transformer": normalization_transformer,
    "oversampler": oversampler,
    "undersampler": undersampler
}
joblib.dump(pipeline_config, OUTPUT_DIR / "pipeline_config.pkl")