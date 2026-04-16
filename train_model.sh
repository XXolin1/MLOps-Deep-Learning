#!/bin/bash

# Configuration: defaults can be overridden by env variables
DATA_DIR=${DATA_DIR:-".cache/data/"}
MODEL_OUT=${MODEL_OUTPUT_DIR:-".cache/model/"}

# Run with CLI parameters (takes precedence over env variables if provided as arguments)
python -m src.model_training.main --data-dir "$DATA_DIR" --model-out "$MODEL_OUT" "$@"
