@echo off
title Audio-to-Video Sync Studio
echo ========================================================
echo       Starting Audio-to-Video Sync Studio
echo ========================================================

cd /d "%~dp0"

IF EXIST "venv\Scripts\activate.bat" (
    echo [INFO] Activating local virtual environment...
    call "venv\Scripts\activate.bat"
) ELSE IF EXIST "..\text.to.video\venv\Scripts\activate.bat" (
    echo [INFO] Activating shared virtual environment (..\text.to.video\venv)...
    call "..\text.to.video\venv\Scripts\activate.bat"
) ELSE (
    echo [WARNING] No virtual environment found. Using system Python...
)

echo [INFO] Launching Web UI at http://127.0.0.1:7861 ...
python app.py

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Application terminated with error code %ERRORLEVEL%
    pause
)
