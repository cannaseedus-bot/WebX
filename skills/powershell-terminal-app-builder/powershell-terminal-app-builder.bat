@echo off
setlocal

if "%~1"=="" (
    echo Usage: powershell-terminal-app-builder.bat ^<chat^|integration^|ide^>
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch_ps_builder.ps1" -Mode "%~1"

endlocal
