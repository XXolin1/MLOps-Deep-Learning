 --epochs "$EPOCHS"@echo off
setlocal

:: Configuration: defaults can be overridden by env variables
if "%DATA_DIR%"=="" set DATA_DIR=.cache/data/
if "%MODEL_OUTPUT_DIR%"=="" set MODEL_OUTPUT_DIR=.cache/model/

python -m src.model_training.main --data-dir "%DATA_DIR%" --model-out "%MODEL_OUTPUT_DIR%" %*

endlocal
