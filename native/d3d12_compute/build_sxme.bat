@echo off
REM ============================================================================
REM Build SCX-MoE DirectX 12 Compute Engine
REM ============================================================================
REM
REM This script builds the DirectX 12 compute shader and C++ wrapper for
REM GPU-accelerated SCX-MoE forward pass.
REM
REM OUTPUTS:
REM   - build\Release\sxme_compute.dll    (Shared library)
REM   - build\Release\d3d12_compute_demo.exe  (Demo executable)
REM
REM REQUIREMENTS:
REM   - Windows 10+ with Visual Studio Build Tools
REM   - Windows SDK 10.0+
REM   - CMake 3.20+
REM
REM USAGE:
REM   .\build_sxme.bat              (Build Release)
REM   .\build_sxme.bat debug        (Build Debug)
REM   .\build_sxme.bat clean        (Clean build artifacts)
REM ============================================================================

setlocal enabledelayedexpansion

REM Colors for output
set RESET=[0m
set GREEN=[92m
set YELLOW=[93m
set RED=[91m
set CYAN=[96m

REM Get command argument
set BUILD_TYPE=%1
if "!BUILD_TYPE!"=="" set BUILD_TYPE=Release
if "!BUILD_TYPE!"=="debug" set BUILD_TYPE=Debug
if "!BUILD_TYPE!"=="clean" goto :clean

echo.
echo %CYAN%========== SCX-MoE DirectX 12 Build System ==========%RESET%
echo.
echo Build type: %GREEN%!BUILD_TYPE!%RESET%
echo Build directory: %CYAN%!CD!\build%RESET%
echo.

REM Check prerequisites
echo %CYAN%[*] Checking prerequisites...%RESET%
where cmake >nul 2>&1
if errorlevel 1 (
    echo %RED%[FAIL] CMake not found. Install from https://cmake.org%RESET%
    exit /b 1
)
echo %GREEN%[OK] CMake found%RESET%

REM Check for MSBuild in two possible locations
set MSBUILD_PATH=
if exist "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\MSBuild\Current\Bin\MSBuild.exe" (
    set "MSBUILD_PATH=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe" (
    set "MSBUILD_PATH=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
)

if "!MSBUILD_PATH!"=="" (
    echo %RED%[FAIL] MSBuild not found. Install Visual Studio Build Tools%RESET%
    exit /b 1
)
echo %GREEN%[OK] MSBuild found at !MSBUILD_PATH!%RESET%

echo.

REM Create build directory
if not exist build (
    echo %CYAN%[*] Creating build directory...%RESET%
    mkdir build
    echo %GREEN%[OK] Build directory created%RESET%
) else (
    echo %CYAN%[*] Using existing build directory%RESET%
)

echo.

REM Configure with CMake
echo %CYAN%[*] Configuring with CMake...%RESET%
cd build
cmake -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=!BUILD_TYPE! ..
if errorlevel 1 (
    echo %RED%[FAIL] CMake configuration failed%RESET%
    cd ..
    exit /b 1
)
echo %GREEN%[OK] CMake configuration complete%RESET%

echo.

REM Build with MSBuild
echo %CYAN%[*] Building with MSBuild...%RESET%
"!MSBUILD_PATH!" ALL_BUILD.vcxproj /p:Configuration=!BUILD_TYPE! /p:Platform=x64 /m:4
if errorlevel 1 (
    echo %RED%[FAIL] MSBuild failed%RESET%
    cd ..
    exit /b 1
)
echo %GREEN%[OK] MSBuild complete%RESET%

cd ..

echo.
echo %CYAN%[*] Build artifacts:%RESET%
if exist "build\!BUILD_TYPE!\sxme_compute.dll" (
    echo %GREEN%[OK]%RESET% build\!BUILD_TYPE!\sxme_compute.dll
) else (
    echo %RED%[MISSING]%RESET% sxme_compute.dll
)

if exist "build\!BUILD_TYPE!\d3d12_compute_demo.exe" (
    echo %GREEN%[OK]%RESET% build\!BUILD_TYPE!\d3d12_compute_demo.exe
) else (
    echo %RED%[MISSING]%RESET% d3d12_compute_demo.exe
)

echo.
echo %GREEN%========== Build Complete ==========%RESET%
echo.
echo Next steps:
echo   1. Test demo: build\!BUILD_TYPE!\d3d12_compute_demo.exe
echo   2. Enable in training: python train_real_full.py --use-directx
echo.

exit /b 0

REM ============================================================================
REM CLEAN: Remove build artifacts
REM ============================================================================
:clean
echo %CYAN%[*] Cleaning build artifacts...%RESET%
if exist build (
    rmdir /s /q build
    echo %GREEN%[OK] Build directory removed%RESET%
) else (
    echo %YELLOW%[WARN] Build directory does not exist%RESET%
)
exit /b 0
