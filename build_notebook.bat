@echo off
rem Rebuild both notebooks from colab_runtime.py and translations.json. Double-click to run.
setlocal
cd /d "%~dp0"
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" build_notebook.py
echo.
if errorlevel 1 (
    echo Build failed. See the message above.
) else (
    echo Done.
)
pause
