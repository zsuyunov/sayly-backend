@echo off
echo Starting Backend Server...
echo.
echo Make sure you're in the backend directory and have activated the virtual environment.
echo.
cd /d %~dp0
call venv\Scripts\activate.bat
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
pause

