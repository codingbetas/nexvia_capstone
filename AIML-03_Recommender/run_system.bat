@echo off
TITLE AIML-03 RecSys Launcher
echo ==============================================================================
echo        AIML-03: Large-Scale Personalized Recommendation Engine
echo                 Two-Stage Retrieval & Precision Ranking
echo ==============================================================================
echo.

cd /d "%~dp0"

echo [1/3] Verifying Python Environment...
if not exist ".\venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\venv.
    echo Please make sure the venv is created.
    pause
    exit /b 1
)

echo [2/3] Running Automated Test Suite...
.\venv\Scripts\python.exe -m unittest tests\test_pipeline.py
if errorlevel 1 (
    echo [ERROR] Unit tests failed!
    pause
    exit /b 1
)
echo [OK] Pipeline tests passed cleanly!
echo.

echo [3/3] Launching Microservices...
echo - Starting FastAPI Server on http://127.0.0.1:8000 ...
start "Nexvia RecSys Backend (FastAPI)" cmd /k ".\venv\Scripts\uvicorn.exe src.api:app --reload --port 8000"

echo - Waiting for backend socket to initialize...
timeout /t 2 /nobreak >nul

echo - Starting Streamlit Dashboard on http://localhost:8501 ...
start "Nexvia RecSys UI (Streamlit)" cmd /k ".\venv\Scripts\streamlit.exe run dashboard.py"

echo.
echo ==============================================================================
echo  Both services launched successfully!
echo.
echo  - Interactive UI:       http://localhost:8501
echo  - Swagger API Docs:     http://127.0.0.1:8000/docs
echo  - Health Endpoint:      http://127.0.0.1:8000/health
echo ==============================================================================
