@echo off
setlocal

:: Default values for Fast API configuration
if "%API_PORT%"=="" set API_PORT=8000
if "%API_HOST%"=="" set API_HOST=0.0.0.0

:: Using uvicorn to run the app
python -m uvicorn src.fastAPI.app:app --host %API_HOST% --port %API_PORT% --reload

endlocal
