@echo off
setlocal

:: Configuration: defaults can be overridden by env variables
if "%DATA_INPUT_PATH%"=="" set DATA_INPUT_PATH=data/diabetes_012_health_indicators_BRFSS2015.csv
if "%DATA_OUTPUT_DIR%"=="" set DATA_OUTPUT_DIR=.cache/data/

python -m src.data_preprocessing.main --input "%DATA_INPUT_PATH%" --output "%DATA_OUTPUT_DIR%" %*

endlocal
