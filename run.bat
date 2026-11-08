@echo off
title BAUST AI Help Desk Server
echo ======================================================
echo    Starting BAUST AI Help Desk Server...
echo ======================================================

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment 'venv' not found!
    echo Please make sure the venv folder exists.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting Flask application...
echo Opening http://127.0.0.1:5000 in your browser...
start http://127.0.0.1:5000

venv\Scripts\python.exe app.py
pause
