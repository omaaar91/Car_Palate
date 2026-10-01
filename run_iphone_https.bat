@echo off
title Egyptian ALPR - Native HTTPS for iPhone
echo ===================================================================
echo   Starting Streamlit with Apple iOS-Compliant HTTPS for iPhone
echo ===================================================================
echo.
echo >>> Open this EXACT link in Safari on your iPhone: <<<
echo.
echo       https://192.168.1.6:8501
echo.
echo -------------------------------------------------------------------
echo Instructions for Safari on iPhone:
echo 1. Type https://192.168.1.6:8501 in Safari.
echo 2. Tap "Show Details" (إظهار التفاصيل).
echo 3. Tap "visit this website" (متابعة إلى هذا الموقع).
echo 4. Tap "Allow" (سماح) when Safari asks for Camera access.
echo ===================================================================
echo.
"E:\car_palate_venv\Scripts\python.exe" -m streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.sslCertFile cert.pem --server.sslKeyFile key.pem
pause
