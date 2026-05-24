@echo off
setlocal

echo.
echo ==========================================
echo    CD1 Finance - Backend + Frontend
echo ==========================================
echo.

if not exist "backend" (
    echo ERROR: backend directory not found.
    exit /b 1
)

if not exist "frontend\package.json" (
    echo ERROR: frontend package.json not found.
    exit /b 1
)

if not exist "backend\uploads" mkdir backend\uploads
if not exist "backend\results" mkdir backend\results
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    python -m venv .venv
)

echo Installing backend dependencies into .venv...
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

if not exist "frontend\node_modules" (
    echo Installing frontend dependencies...
    pushd frontend
    call npm install
    popd
)

echo Starting backend at http://localhost:5000
start "CD1 Backend" cmd /k "cd /d %CD%\backend && %CD%\.venv\Scripts\python.exe -m scripts.serve"

timeout /t 3 /nobreak > nul

echo Starting frontend at http://localhost:3000
start "CD1 Frontend" cmd /k "cd /d %CD%\frontend && npm start"

echo.
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:3000
echo.
pause
