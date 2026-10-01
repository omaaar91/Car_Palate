@echo off
title Egyptian ALPR - FastAPI Backend
echo ========================================================
echo   Starting Egyptian ALPR FastAPI Backend
echo   API Docs: http://localhost:8000/docs
echo ========================================================
"E:\car_palate_venv\Scripts\python.exe" -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
pause
