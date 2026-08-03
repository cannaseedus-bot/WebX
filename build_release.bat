@echo off
REM ============================================================================
REM K'UHUL Engine Build Script - Windows x64 (Visual Studio 2022)
REM ============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%build"
set "BIN_DIR=%BUILD_DIR%\bin"
set "CMAKE_EXE=C:\Program Files\CMake\bin\cmake.exe"

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║         K'UHUL ENGINE BUILD (Visual Studio 2022)           ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM ============================================================================
REM Check Prerequisites
REM ============================================================================

echo [1/5] Checking prerequisites...
echo.

if not exist "!CMAKE_EXE!" (
    echo ❌ CMake not found at: !CMAKE_EXE!
    echo    Please install CMake from: https://cmake.org/download/
    exit /b 1
)

REM Try to locate Visual Studio 2022 vcvars64.bat
set "VCVARS64="
for %%p in (
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) do (
    if exist "%%~p" (
        set "VCVARS64=%%~p"
        goto :vcvars_found
    )
)

:vcvars_found
where cl >nul 2>&1
if errorlevel 1 (
    if defined VCVARS64 (
        echo Setting up Visual Studio environment from: !VCVARS64!
        call "!VCVARS64!" >nul
    ) else (
        echo ❌ Visual C++ compiler not found and no vcvars64.bat detected.
        echo    Please run from "x64 Native Tools Command Prompt for VS 2022"
        echo    Or install Visual Studio 2022 with "Desktop development with C++"
        exit /b 1
    )
)

where cl >nul 2>&1
if errorlevel 1 (
    echo ❌ Visual C++ compiler still not found after vcvars64.bat
    exit /b 1
)

echo ✓ CMake found: !CMAKE_EXE!
echo ✓ Visual C++ compiler found
echo.

REM ============================================================================
REM Create Build Directory
REM ============================================================================

echo [2/5] Creating build directory...
echo.

if exist "!BUILD_DIR!" (
    echo  Cleaning previous build...
    rmdir /s /q "!BUILD_DIR!" >nul 2>&1
)

mkdir "!BUILD_DIR!"
if errorlevel 1 (
    echo ❌ Failed to create build directory
    exit /b 1
)

echo ✓ Build directory created: !BUILD_DIR!
echo.

REM ============================================================================
REM Configure with CMake
REM ============================================================================

echo [3/5] Configuring with CMake...
echo.

cd /d "!BUILD_DIR!"

"!CMAKE_EXE!" .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_INSTALL_PREFIX="!BIN_DIR!"

if errorlevel 1 (
    echo ❌ CMake configuration failed
    cd /d "!SCRIPT_DIR!"
    exit /b 1
)

echo ✓ Configuration complete
echo.

REM ============================================================================
REM Build
REM ============================================================================

echo [4/5] Building kuhul_engine.exe (Release)...
echo.

"!CMAKE_EXE!" --build . --config Release --parallel

if errorlevel 1 (
    echo ❌ Build failed
    cd /d "!SCRIPT_DIR!"
    exit /b 1
)

echo ✓ Build complete
echo.

REM ============================================================================
REM Copy Runtime Dependencies
REM ============================================================================

echo [4b/5] Copying runtime dependencies...
echo.

if exist "!SCRIPT_DIR!native\d3d12_compute\build\Release\dxcompiler.dll" (
    copy /y "!SCRIPT_DIR!native\d3d12_compute\build\Release\dxcompiler.dll" "!BIN_DIR!\Release\" >nul
    echo ✓ Copied dxcompiler.dll
)

echo.

REM ============================================================================
REM Verify Output
REM ============================================================================

echo [5/5] Verifying build output...
echo.

if exist "!BIN_DIR!\Release\kuhul_engine.exe" (
    for /f "delims=" %%a in ('dir /b "!BIN_DIR!\Release\kuhul_engine.exe"') do (
        set "filename=%%a"
    )
    
    echo ✓ Binary created: !filename!
    
    REM Get file size
    for %%A in ("!BIN_DIR!\Release\kuhul_engine.exe") do (
        set "size=%%~zA"
        set /a size_mb=!size! / 1048576
        echo   Size: !size_mb! MB
    )
    
    echo.
    echo ╔════════════════════════════════════════════════════════════╗
    echo ║              BUILD SUCCESSFUL ✓                            ║
    echo ╠════════════════════════════════════════════════════════════╣
    echo ║ Output:  !BIN_DIR!\Release\kuhul_engine.exe              
    echo ║ Run:     kuhul_engine.exe --help                          ║
    echo ║ Docs:    .github/README.md                                ║
    echo ╚════════════════════════════════════════════════════════════╝
    echo.
    
    cd /d "!SCRIPT_DIR!"
    exit /b 0
) else (
    echo ❌ Binary not found at expected location
    echo    Expected: !BIN_DIR!\Release\kuhul_engine.exe
    cd /d "!SCRIPT_DIR!"
    exit /b 1
)

endlocal
