@echo off
setlocal EnableDelayedExpansion

set ENV_NAME=churcol-deploy
set PROJECT=C:\Users\lulay\Desktop\kewirausahaan-tech\qr-app

echo.
echo ============================================================
echo  CHURCOL -- Clean environment deploy test
echo ============================================================
echo  Environment : %ENV_NAME%
echo  Python      : 3.11
echo  Project     : %PROJECT%
echo ============================================================
echo.

:: ── Step 1: create env ────────────────────────────────────────
echo [1/4] Creating conda environment %ENV_NAME% with Python 3.11...
call conda create -n %ENV_NAME% python=3.11 -y
if errorlevel 1 ( echo ERROR: conda create failed & exit /b 1 )

:: ── Activate the new env for all subsequent commands ──────────
call conda activate %ENV_NAME%
if errorlevel 1 ( echo ERROR: conda activate failed & exit /b 1 )

:: ── Step 2: install project requirements ─────────────────────
echo.
echo [2/4] Installing requirements.txt...
pip install -r "%PROJECT%\requirements.txt"
if errorlevel 1 ( echo ERROR: pip install requirements failed & exit /b 1 )

:: ── Step 3: install pytest ────────────────────────────────────
echo.
echo [3/4] Installing pytest...
pip install pytest
if errorlevel 1 ( echo ERROR: pip install pytest failed & exit /b 1 )

:: ── Step 4: run deployment tests ─────────────────────────────
echo.
echo [4/4] Running deployment tests...
cd /d "%PROJECT%"
python -m pytest test/test_deployment.py -v
if errorlevel 1 (
    echo.
    echo RESULT: Some tests failed -- fix before deploying.
    exit /b 1
)

echo.
echo ============================================================
echo  All tests passed. Safe to deploy to Streamlit Cloud.
echo ============================================================
endlocal
