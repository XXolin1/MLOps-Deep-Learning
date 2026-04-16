#!/bin/bash

# Configuration: defaults can be overridden by env variables
INPUT_PATH=${DATA_INPUT_PATH:-"data/diabetes_012_health_indicators_BRFSS2015.csv"}
OUTPUT_DIR=${DATA_OUTPUT_DIR:-".cache/data/"}

# Run with CLI parameters (takes precedence over env variables if provided as arguments)
python -m src.data_preprocessing.main --input "$INPUT_PATH" --output "$OUTPUT_DIR" "$@"
