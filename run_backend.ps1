# Launch Concealed AI Obfuscation API Backend
Write-Host "Starting Concealed AI Backend Server on http://127.0.0.1:8001..." -ForegroundColor Cyan
Write-Host "Interactive Swagger Docs: http://127.0.0.1:8001/docs" -ForegroundColor Green
python -m concealed.api.main --host 127.0.0.1 --port 8001 --reload
