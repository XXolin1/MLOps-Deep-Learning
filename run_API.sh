#!/bin/bash

# Default values for FastAPI configuration
PORT=${API_PORT:-"8000"}
HOST=${API_HOST:-"0.0.0.0"}

# Default cache paths
MODEL_CACHE=${MODEL_CACHE:-".cache/model"}
DATA_CACHE=${DATA_CACHE:-".cache/data"}

# Parse optional command-line arguments for cache paths
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-cache)
            MODEL_CACHE="$2"
            shift 2
            ;;
        --data-cache)
            DATA_CACHE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model-cache PATH] [--data-cache PATH] [--port PORT] [--host HOST]"
            exit 1
            ;;
    esac
done

# Export environment variables for the CLI parameter parser
export MODEL_CACHE
export DATA_CACHE

echo "FastAPI Configuration:"
echo "  HOST: $HOST"
echo "  PORT: $PORT"
echo "  MODEL_CACHE: $MODEL_CACHE"
echo "  DATA_CACHE: $DATA_CACHE"

# Using uvicorn to run the app with cache parameters passed via environment variables
python -m uvicorn src.fastAPI.app:app --host "$HOST" --port "$PORT" --reload
