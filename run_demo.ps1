# CV-FL Interactive Demo Launcher (PowerShell)
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " Launching CV-FL Privacy-Preserving Federated Learning Demo" -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment (venv)..." -ForegroundColor Green
    & ".\venv\Scripts\Activate.ps1"
}

python demo.py
