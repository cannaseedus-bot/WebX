@echo off
REM ============================================================================
REM Build SCX-MoE DirectX 12 Compute Engine (Ninja variant)
REM ============================================================================
REM Uses Ninja build system instead of MSBuild
REM Install: winget install Ninja-build
REM ============================================================================

setlocal enabledelayedexpansion

REM Colors
set GREEN=[92m
set YELLOW=[93m
set RED=[91m
set CYAN=[96m
set RESET=[0m

set BUILD_TYPE=%1
if "!BUILD_TYPE!"=="" set BUILD_TYPE=Release
if "!BUILD_TYPE!"=="debug" set BUILD_TYPE=Debug
if "!BUILD_TYPE!"=="clean" goto :clean

echo.
echo %CYAN%========== SCX-MoE DirectX 12 Build (Ninja) ==========%RESET%
echo Build type: %GREEN%!BUILD_TYPE!%RESET%
echo.

REM Check prerequisites
where ninja >nul 2>&1
if errorlevel 1 (
    echo %RED%[FAIL] Ninja not found%RESET%
    echo.
    echo Install with: winget install Ninja-build
    exit /b 1
)
echo %GREEN%[OK] Ninja found%RESET%

where cmake >nul 2>&1
if errorlevel 1 (
    echo %RED%[FAIL] CMake not found%RESET%
    exit /b 1
)
echo %GREEN%[OK] CMake found%RESET%

echo.

REM Create/clean build directory
if not exist build (
    mkdir build
    echo %CYAN%[*] Created build directory%RESET%
)

echo.
echo %CYAN%[*] Configuring with CMake (Ninja)...%RESET%
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=!BUILD_TYPE! ..
if errorlevel 1 (
    echo %RED%[FAIL] CMake failed%RESET%
    cd ..
    exit /b 1
)
echo %GREEN%[OK] CMake complete%RESET%

echo.
echo %CYAN%[*] Building with Ninja...%RESET%
ninja -j 4
if errorlevel 1 (
    echo %RED%[FAIL] Ninja build failed%RESET%
    cd ..
    exit /b 1
)
echo %GREEN%[OK] Build complete%RESET%

cd ..

echo.
echo %CYAN%[*] Output files:%RESET%
if exist "build\sxme_compute.dll" (
    echo %GREEN%[OK]%RESET% build\sxme_compute.dll
) else (
    echo %RED%[MISSING]%RESET% sxme_compute.dll
)

if exist "build\d3d12_compute_demo.exe" (
    echo %GREEN%[OK]%RESET% build\d3d12_compute_demo.exe
) else (
    echo %RED%[MISSING]%RESET% d3d12_compute_demo.exe
)

echo.
echo %GREEN%========== Build Complete ==========%RESET%
echo.

exit /b 0

:clean
echo %CYAN%[*] Cleaning...%RESET%
if exist build (
    rmdir /s /q build
    echo %GREEN%[OK] Cleaned%RESET%
)
exit /b 0
