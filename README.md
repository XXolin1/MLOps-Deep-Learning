# MLOps Deep Learning - Diabetes Prediction

A production-ready MLOps pipeline for diabetes prediction using deep learning. This project demonstrates best practices for data preprocessing, model training, and API deployment with containerization.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [API Usage](#api-usage)
- [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)

## Project Overview

This project implements a complete ML pipeline for diabetes prediction:

1. **Data Preparation**: Clean, normalize, and resample data with SMOTE and NearMiss
2. **Model Training**: Train deep learning models with optimized thresholds
3. **FastAPI Service**: Production-ready REST API for predictions
4. **Docker Deployment**: Containerized application for easy deployment

### Key Features

- Robust data preprocessing pipeline
- SMOTE oversampling & NearMiss undersampling for balanced data
- FastAPI with automatic documentation
- CLI parameter support for flexible configuration
- Docker & Docker Compose for containerization
- Health checks and error handling
- Modular, maintainable code structure

## Project Structure

```
MLOps-Deep-Learning/
├── src/
│   ├── data_preprocessing/
│   │   ├── main.py                 # Data generation pipeline
│   │   ├── processing_class.py    # Custom transformers & pipelines
│   ├── model_training/
│   │   ├── main.py                 # Model training script
│   │   ├── model_class.py          # Model definitions
│   ├── fastAPI/
│   │   ├── app.py                  # FastAPI application
│   │   ├── predict.py              # Prediction endpoint logic
│   │   ├── metrics.py              # Metrics calculation
│   │   ├── cache_loading.py        # Cache loading utilities
│   │   ├── cli_parameters.py       # CLI parameter parsing
│   ├── models/                     # Trained models directory
├── data/                           # Raw data directory
├── .cache/
│   ├── models/                     # Cached model artifacts
│   ├── data/                       # Cached preprocessed data
├── generate_data.sh                # Data generation script (Linux/Mac)
├── generate_data.bat               # Data generation script (Windows)
├── train_model.sh                  # Training script (Linux/Mac)
├── train_model.bat                 # Training script (Windows)
├── run_API.sh                      # API startup script (Linux/Mac)
├── run_API.bat                     # API startup script (Windows)
├── Dockerfile                      # Docker image definition
├── docker-compose.yml              # Docker Compose configuration
├── requirements.txt                # Python dependencies
├── README.md                       # This file
```

## Prerequisites

- **Python**: 3.10+
- **Docker**: 20.10+ (optional, for containerized deployment)
- **Docker Compose**: 1.29+ (optional)
- **Dataset**: `diabetes_012_health_indicators_BRFSS2015.csv` in the `data/` directory

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd MLOps-Deep-Learning
```

### 2. Create Virtual Environment

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Data Preparation

### Generate Preprocessed Data

The data preparation pipeline includes:

- Feature dropping and missing value imputation
- BMI categorization
- Deduplication
- Normalization (MinMaxScaler, StandardScaler, etc.)
- Stratified train/validation/test split
- SMOTE oversampling (60% minority class)
- NearMiss undersampling (70% majority class)

#### Using Shell Script (Linux/Mac)

```bash
bash generate_data.sh
```

#### Using Batch Script (Windows)

```cmd
generate_data.bat
```

#### Using Python Directly

```bash
python -m src.data_preprocessing.main \
  --input dataset/diabetes_012_health_indicators_BRFSS2015.csv \
  --output data
```

#### With Environment Variables

```bash
export DATA_INPUT_PATH="dataset/diabetes_012_health_indicators_BRFSS2015.csv"
export DATA_OUTPUT_DIR="data"
python -m src.data_preprocessing.main
```

**Output files** in `data/` directory:

- `train_data.csv` - Training data (resampled)
- `validation_data.csv` - Validation data
- `test_data.csv` - Test data
- `categorize_bmi.joblib` - BMI categorizer transformer
- `normalization_transformer.joblib` - Normalization pipeline
- `feature_dropper.joblib` - Feature dropper transformer

## Model Training

### Train the Model

The training pipeline includes:

- Model architecture definition
- Optimal threshold calculation
- Model evaluation on test data

#### Using Shell Script (Linux/Mac)

```bash
bash train_model.sh
```

#### Using Batch Script (Windows)

```cmd
train_model.bat
```

#### Using Python Directly

```bash
python -m src.model_training.main \
  --data-cache data \
  --model-cache models
```

**Output files** in `models/` directory:

- `model.joblib` - Trained model
- `threshold.joblib` - Optimal classification threshold

## API Usage

### Local Development

#### Using Shell Script (Linux/Mac)

```bash
bash run_API.sh
```

#### Using Batch Script (Windows)

```cmd
run_API.bat
```

#### Using Python Directly

```bash
python -m uvicorn src.fastAPI.app:app --host 0.0.0.0 --port 8000 --reload
```

#### With Custom Cache Paths

```bash
./run_API.sh --model-cache ./custom_models --data-cache ./custom_data --port 8080 --host 127.0.0.1
```

#### With Environment Variables

```bash
export MODEL_CACHE=./models
export DATA_CACHE=./data
bash run_API.sh
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Docker Deployment

### Build and Run with Docker Compose

```bash
# Start the container
docker-compose up -d

# View logs
docker-compose logs -f fastapi

# Stop the container
docker-compose down
```

### Build Manually

```bash
# Build the image
docker build -t mlops-fastapi .

# Run the container
docker run -d \
  --name mlops-fastapi \
  -p 8000:8000 \
  -v $(pwd)/.cache:/app/.cache:ro \
  -e MODEL_CACHE=/app/.cache/models \
  -e DATA_CACHE=/app/.cache/data \
  mlops-fastapi
```

### Access the Container

```bash
# Check container status
docker ps

# View logs
docker logs mlops-fastapi

# Access the API
curl http://localhost:8000/
```

## Configuration

### CLI Parameters

The FastAPI application supports the following CLI parameters:

```bash
python -m uvicorn src.fastAPI.app:app --help
```

#### Model Cache Configuration

```bash
./run_API.sh --model-cache /path/to/models
```

**Environment Variable**: `MODEL_CACHE`

Default: `.cache/models`

#### Data Cache Configuration

```bash
./run_API.sh --data-cache /path/to/data
```

**Environment Variable**: `DATA_CACHE`

Default: `.cache/data`

#### Port Configuration

```bash
./run_API.sh --port 8080
```

**Environment Variable**: `API_PORT`

Default: `8000`

#### Host Configuration

```bash
./run_API.sh --host 127.0.0.1
```

**Environment Variable**: `API_HOST`

Default: `0.0.0.0`

## API Endpoints

### Root Endpoint

**GET** `/`

Returns a welcome message.

```bash
curl http://localhost:8000/
```

**Response:**

```json
{ "message": "Welcome in our deep Learning project!" }
```

### Prediction Endpoint

**POST** `/predict`

Make predictions on new data.

**Request Body:**

```json
{
  "HighBP": 1,
  "HighChol": 0,
  "CholCheck": 1,
  "BMI": 25.5,
  "Smoker": 0,
  "Stroke": 0,
  "HeartDiseaseorAttack": 0,
  "PhysActivity": 1,
  "Fruits": 0,
  "Veggies": 1,
  "HvyAlcoholConsump": 0,
  "AnyHealthcare": 1,
  "NoDocbcCost": 0,
  "GenHlth": 2,
  "MentHlth": 0,
  "PhysHlth": 0,
  "DiffWalk": 0,
  "Sex": 1,
  "Age": 5,
  "Education": 4,
  "Income": 3
}
```

**Response:**

```json
{
  "prediction": 1,
  "probability": 0.75,
  "threshold": 0.5
}
```

### Metrics Endpoint

**POST** `/metrics`

Calculate metrics on test data (requires test data to be available).

**Response:**

```json
{
  "accuracy": 0.85,
  "precision": 0.82,
  "recall": 0.88,
  "f1": 0.85,
  "confusion_matrix": [
    [1200, 150],
    [100, 550]
  ]
}
```

## 📝 Environment Variables Summary

| Variable          | Default                                                | Description                         |
| ----------------- | ------------------------------------------------------ | ----------------------------------- |
| `MODEL_CACHE`     | `.cache/models`                                        | Path to cached model artifacts      |
| `DATA_CACHE`      | `.cache/data`                                          | Path to cached preprocessed data    |
| `API_PORT`        | `8000`                                                 | FastAPI server port                 |
| `API_HOST`        | `0.0.0.0`                                              | FastAPI server host                 |
| `DATA_INPUT_PATH` | `dataset/diabetes_012_health_indicators_BRFSS2015.csv` | Path to raw dataset                 |
| `DATA_OUTPUT_DIR` | `data`                                                 | Output directory for processed data |

## Troubleshooting

### Model Not Found

**Error**: `FileNotFoundError: Model not found at .cache/models/model.joblib`

**Solution**: Ensure you've run the training pipeline first:

```bash
bash generate_data.sh
bash train_model.sh
```

### Data Not Found

**Error**: `FileNotFoundError: Normalization transformer not found`

**Solution**: Generate the data first:

```bash
bash generate_data.sh
```

### Cache Directory Issues

**Error**: `Warning: Model cache directory does not exist`

**Solution**: Create the cache directories manually:

```bash
mkdir -p .cache/models
mkdir -p .cache/data
```

### Port Already in Use

**Error**: `Address already in use` on port 8000

**Solution**: Use a different port:

```bash
./run_API.sh --port 8080
```

## Dependencies

Main dependencies (see `requirements.txt` for full list):

- **FastAPI**: Web framework for building APIs
- **Uvicorn**: ASGI server
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Imbalanced-learn**: SMOTE/NearMiss resampling
- **PyTorch**: Deep learning framework (for models)
- **Joblib**: Serialization of models and pipelines

## License

This project is part of the CESI AI & Deep Learning curriculum.

## Contributors

Guillaume - MLOps Pipeline Development

## Support

For issues or questions, please refer to the project documentation or contact the development team.
