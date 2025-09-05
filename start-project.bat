@echo off
echo Starting REIMAGINE AI Project...
echo.

echo Starting Backend Server (Simple Mode - No Database Required)...
start "Backend Server" cmd /k "cd backend && node server-simple.js"

timeout /t 3 /nobreak > nul

echo Starting Frontend Development Server...
start "Frontend Server" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers are starting up...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:5173
echo.
echo Your REIMAGINE AI project is now running!
echo Open http://localhost:5173 in your browser to see the application.
echo.
echo Press any key to exit this window...
pause > nul
