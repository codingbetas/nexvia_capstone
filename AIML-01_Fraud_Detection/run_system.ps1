# AIML-01 Fraud Detection Platform Launcher (PowerShell)
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "       AIML-01: Real-Time Financial Fraud Detection Platform" -ForegroundColor Cyan
Write-Host "                FastAPI Microservice & Analyst Dashboard" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ""

$workDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $workDir

Write-Host "[1/2] Verifying Python Environment..." -ForegroundColor Yellow
if (!(Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "[ERROR] Virtual environment not found at .\venv." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Python environment verified!" -ForegroundColor Green

Write-Host ""
Write-Host "[2/2] Launching Microservices..." -ForegroundColor Yellow
Write-Host "- Starting FastAPI on http://127.0.0.1:8000..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$workDir'; .\venv\Scripts\uvicorn.exe main:app --reload --port 8000"

Start-Sleep -Seconds 2

Write-Host "- Starting Streamlit on http://localhost:8501..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$workDir'; .\venv\Scripts\streamlit.exe run dashboard.py"

Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Green
Write-Host " Both services launched successfully!" -ForegroundColor Green
Write-Host " - Interactive UI:   http://localhost:8501" -ForegroundColor White
Write-Host " - Swagger API Docs: http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "==============================================================================" -ForegroundColor Green
