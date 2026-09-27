@echo off
echo Starting Concealed AI Backend Server...
python -m concealed.api.main --host 127.0.0.1 --port 8001 --reload
pause
