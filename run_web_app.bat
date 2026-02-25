@echo off
rem Change working directory to the script's location
cd /d "%~dp0"

echo Installing dependencies...
pip install fastapi uvicorn jinja2 python-dotenv requests

echo.
echo Starting Web UI...
python web_app.py

pause
