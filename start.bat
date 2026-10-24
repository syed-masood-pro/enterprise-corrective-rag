@echo off
setlocal enabledelayedexpansion
title RAG Application Launcher
color 0A

echo ========================================
echo    RAG Document Search - Starting...
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: .\venv\Scripts\Activate.ps1
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

echo [1/3] Starting Backend API (FastAPI + Uvicorn)...

REM Kill anything already using port 8000 or 3000
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000.*LISTENING"') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":3000.*LISTENING"') do taskkill /PID %%a /F >nul 2>&1
timeout /t 2 /nobreak >nul

REM Start backend in the background
start /B "RAG Backend" cmd /c "venv\Scripts\activate && uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1"
echo Loading embedding model and building search index...
echo This may take several minutes with large document collections...

:backend_check
timeout /t 5 /nobreak >nul
set STATUS=000
for /f %%i in ('curl -s -o nul -w "%%{http_code}" --max-time 2 http://localhost:8000/health') do set STATUS=%%i

if "!STATUS!"=="200" goto backend_ready

<nul set /p="."
goto backend_check

:backend_ready
echo.
echo Backend ready!

echo [2/3] Starting Frontend Server (Port 3000)...
start /B "RAG Frontend" cmd /c "cd frontend && python -m http.server 3000"

echo Checking frontend...
timeout /t 3 /nobreak >nul
set F_STATUS=000
for /f %%i in ('curl -s -o nul -w "%%{http_code}" --max-time 2 http://localhost:3000') do set F_STATUS=%%i

if "!F_STATUS!"=="200" (
    echo Frontend ready!
) else (
    echo.
    echo [ERROR] Frontend failed to start! ^(HTTP !F_STATUS!^)
    echo Stopping backend...
    for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000.*LISTENING"') do taskkill /PID %%a /F >nul 2>&1
    pause
    exit /b 1
)

echo [3/3] Opening Browser...
start http://localhost:3000

echo.
echo ========================================
echo    Application Running!
echo ========================================
echo Frontend: http://localhost:3000
echo Backend:  http://localhost:8000/docs
echo.
echo Press any key in this terminal to STOP all services and exit.
pause >nul

echo.
echo Stopping services...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000.*LISTENING"') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":3000.*LISTENING"') do taskkill /PID %%a /F >nul 2>&1
echo Done. All services stopped.
exit /b 0