@echo off
setlocal

title KUHUL WebX-3D Launcher

echo.
echo  ^+^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^+
echo  ^|   ^[1m K'UHUL WebX-3D  v3.5.0 ^[0m                              ^|
echo  ^|   D12WebX  ^|  SVG-3D  ^|  7-Brain Topology  ^|  Hybrid Trainer ^|
echo  ^+^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^+
echo.

:: ── working directory ──────────────────────────────────────────────────────────
cd /d "%~dp0"

:: ── check node ────────────────────────────────────────────────────────────────
where node >nul 2>&1
if errorlevel 1 (
  echo  [ERROR] Node.js not found. Install from https://nodejs.org
  pause & exit /b 1
)
for /f "tokens=*" %%v in ('node --version 2^>nul') do set NODE_VER=%%v
echo  [OK] Node.js %NODE_VER%

:: ── start static file server (port 7430) ─────────────────────────────────────
echo  [>>] Starting WebX-3D server on http://127.0.0.1:7430
start "KUHUL-Server" /min cmd /c "node server.cjs"

:: ── optionally start trainer server (port 7431) ──────────────────────────────
set TRAINER_SCRIPT=native\trainer-server.cjs
if exist "%TRAINER_SCRIPT%" (
  echo  [>>] Starting trainer server on http://127.0.0.1:7431
  start "KUHUL-Trainer" /min cmd /c "node %TRAINER_SCRIPT%"
) else (
  echo  [--] Trainer server not found ^(optional^)
)

:: ── brief wait for servers to initialise ─────────────────────────────────────
timeout /t 2 /nobreak >nul

:: ── open browser ─────────────────────────────────────────────────────────────
set URL=http://127.0.0.1:7430
echo  [>>] Opening %URL%
start "" "%URL%"

echo.
echo  [OK] KUHUL WebX-3D is running.
echo.
echo  http://127.0.0.1:7430          Landing page ^& demo shell
echo  http://127.0.0.1:7430/app      3D Runtime ^(src/index.html^)
echo  http://127.0.0.1:7431          Trainer SSE bridge
echo.
echo  Press any key to stop all servers and exit.
pause >nul

:: ── shutdown ──────────────────────────────────────────────────────────────────
echo  Stopping servers...
taskkill /fi "WindowTitle eq KUHUL-Server*"  /f >nul 2>&1
taskkill /fi "WindowTitle eq KUHUL-Trainer*" /f >nul 2>&1
echo  Done.
endlocal
