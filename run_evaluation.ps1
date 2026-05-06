$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   BIS SmartStandards - AI Evaluation     " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Run the inference pipeline
Write-Host "[+] Running inference on the public test set..." -ForegroundColor Yellow
python inference.py --input dataset/public_test_set.json --output dataset/team_results.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "[-] Inference failed!" -ForegroundColor Red
    exit
}

Write-Host "[+] Inference completed successfully." -ForegroundColor Green

# 2. Run the evaluation script
Write-Host "[+] Running evaluation script to generate performance metrics..." -ForegroundColor Yellow
python dataset/eval_script.py --results dataset/team_results.json

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "[+] Evaluation Finished! You can copy the metrics above into your README.md" -ForegroundColor Green
