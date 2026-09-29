@echo off
title Egyptian ALPR - Streamlit Mobile UI
echo ========================================================
echo   Starting Egyptian ALPR Streamlit Mobile UI
echo   Open on Mobile via: Network URL shown below
echo ========================================================
"C:\Program Files\Python311\python.exe" -m streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
pause
