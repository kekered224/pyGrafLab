@echo off
chcp 65001 >nul 2>&1
setlocal
title pyGrafLab Web UI
cd /d "%~dp0"

echo ============================================================
echo   pyGrafLab - Spherical Harmonic Synthesis (Python)
echo   Web Interface Launcher
echo ============================================================
echo.

REM ---- 1. Pick an isolated interpreter -------------------------------------
REM Never run "pip install" against the global / user site-packages: that can
REM downgrade numpy/pillow for unrelated projects. Priority:
REM   1) .venv next to this script   2) shared ..\..\.venv   3) create .venv
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=%CD%\.venv\Scripts\python.exe"
if not defined PY if exist "..\..\.venv\Scripts\python.exe" (
    for %%F in ("..\..\.venv\Scripts\python.exe") do set "PY=%%~fF"
)
if not defined PY (
    echo [INFO] No virtual environment found. Creating ".venv" here...
    python -m venv ".venv"
    if errorlevel 1 (
        echo [ERROR] Could not create a virtual environment.
        echo         Install Python 3.9+ and make sure "python" is on PATH.
        pause
        exit /b 1
    )
    set "PY=%CD%\.venv\Scripts\python.exe"
)
"%PY%" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] The selected Python is broken: "%PY%"
    echo         Delete ".venv" and run this script again.
    pause
    exit /b 1
)
echo [INFO] Python: "%PY%"

REM ---- 2. Check what the server actually needs -----------------------------
REM Importing the real server module is the only reliable success test.
REM pip may print "ERROR: pip's dependency resolver ..." while still
REM succeeding, and may exit non-zero on cleanup while the package is fine.
set "CHECK=import numpy, scipy, flask; import pygraflab.web.server"
"%PY%" -c "%CHECK%" >nul 2>&1
if not errorlevel 1 goto ready

echo [INFO] Missing runtime dependencies. Installing into this environment only...
"%PY%" -m pip install -e ".[web]"
"%PY%" -c "%CHECK%" >nul 2>&1
if not errorlevel 1 goto ready

echo [ERROR] pyGrafLab is still not runnable in this environment.
echo         Run manually:  "%PY%" -m pip install -e ".[web]"
echo         Note: pip dependency-conflict notes are warnings, not failures.
pause
exit /b 1

:ready
echo [OK]   Environment ready (no global packages were modified).
echo.
echo Starting web UI (default http://127.0.0.1:5002); browser opens automatically.
echo Extra arguments are passed through, e.g.  start.bat --no-browser --port 5003
echo Press Ctrl+C to stop the server.
echo.

"%PY%" -m pygraflab.web %*
set "RC=%errorlevel%"
echo.
if not "%RC%"=="0" echo [WARN] Server exited with code %RC%.
pause
exit /b %RC%
