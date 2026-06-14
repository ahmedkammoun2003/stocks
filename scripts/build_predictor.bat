@echo off
REM Build stocks-predictor.exe on Windows (run from project root).
setlocal
cd /d "%~dp0\.."

if not exist "venv\Scripts\python.exe" (
    echo Create a virtual environment first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    exit /b 1
)

venv\Scripts\python.exe scripts\build_predictor.py %*
exit /b %ERRORLEVEL%
