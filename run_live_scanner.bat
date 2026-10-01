@echo off
title Egyptian ALPR - Real-Time Live AR Scanner
echo ===================================================================
echo   Starting Real-Time Live Camera Scanner for Mobile / iPhone
echo ===================================================================
echo.
echo >>> Open this link in Safari on your iPhone: <<<
echo.
echo       https://192.168.1.6:8000/scanner
echo.
echo ===================================================================
"E:\car_palate_venv\Scripts\python.exe" -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --ssl-certfile cert.pem --ssl-keyfile key.pem
pause
