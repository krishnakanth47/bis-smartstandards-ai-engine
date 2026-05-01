$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   BIS SmartStandards AI Engine - Startup  " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check Python version
$pythonVersion = (python --version 2>&1)
Write-Host "[+] Python Version: $pythonVersion"

# Install dependencies if not installed
Write-Host "[+] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[-] Failed to install requirements" -ForegroundColor Red
    exit
}

# Ensure data folder exists
if (-not (Test-Path -Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
    Write-Host "[+] Created data directory for FAISS caching."
}

# Start Server
Write-Host "[+] Starting FastAPI server on port 10000..." -ForegroundColor Green
Write-Host "[!] The server will auto-generate the FAISS index if not present." -ForegroundColor Yellow
Write-Host "    This may take a few minutes for the first run." -ForegroundColor Yellow

$env:PYTHONUNBUFFERED="1"
$env:TOKENIZERS_PARALLELISM="false"

uvicorn app:app --host 0.0.0.0 --port 10000
