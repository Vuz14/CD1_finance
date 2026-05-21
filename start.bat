@echo off
echo.
echo ==========================================
echo    ML Model Trainer - Full Stack App
echo ==========================================
echo.

REM Check if we're in the correct directory
if not exist "frontend" (
    echo ERROR: frontend directory not found!
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

REM Create necessary directories
if not exist "backend\results" mkdir backend\results
if not exist "backend\uploads" mkdir backend\uploads

REM Install frontend dependencies if needed
if not exist "frontend\node_modules" (
    echo.
    echo Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
)

REM Start backend in a new window
echo.
echo Starting Backend Server...
start "Backend Server" cmd /k "cd backend && python app.py"

REM Wait a moment for backend to start
timeout /t 3 /nobreak

REM Start frontend in a new window
echo Starting Frontend Server...
start "Frontend Server" cmd /k "cd frontend && npm start"

echo.
echo ==========================================
echo Services are starting...
echo.
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Close these windows to stop the servers.
echo ==========================================
echo.

pause
