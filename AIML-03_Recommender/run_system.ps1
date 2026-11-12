# AIML-03 Two-Stage Recommendation Engine Launcher (PowerShell)
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "       AIML-03: Large-Scale Personalized Recommendation Engine" -ForegroundColor Cyan
Write-Host "                Two-Stage Retrieval & Precision Ranking" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ""

$workDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $workDir

Write-Host "[1/3] Verifying Python Environment..." -ForegroundColor Yellow
if (!(Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "[ERROR] Virtual environment not found at .\venv." -ForegroundColor Red
    exit 1
}

Write-Host "[2/3] Running Automated Test Suite..." -ForegroundColor Yellow
& ".\venv\Scripts\python.exe" -m unittest tests/test_pipeline.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Tests failed! Aborting startup." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Pipeline tests passed cleanly!" -ForegroundColor Green
Write-Host ""

Write-Host "[3/3] Launching Microservices..." -ForegroundColor Yellow
Write-Host "- Starting FastAPI on http://127.0.0.1:8000..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$workDir'; .\venv\Scripts\uvicorn.exe src.api:app --reload --port 8000"

Start-Sleep -Seconds 2

Write-Host "- Starting Streamlit on http://localhost:8501..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$workDir'; .\venv\Scripts\streamlit.exe run dashboard.py"

Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Green
Write-Host " Both services launched successfully!" -ForegroundColor Green
Write-Host " - Interactive UI:   http://localhost:8501" -ForegroundColor White
Write-Host " - Swagger API Docs: http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "==============================================================================" -ForegroundColor Green
