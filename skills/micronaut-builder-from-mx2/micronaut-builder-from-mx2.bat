@echo off
setlocal

if "%~1"=="" (
    echo Usage: micronaut-builder-from-mx2.bat ^<TargetRoot^>
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build_from_mx2.ps1" -TargetRoot "%~1"

endlocal
