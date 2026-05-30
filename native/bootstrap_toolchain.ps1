#!/usr/bin/env pwsh
#############################################################################
# MX2LM Native Toolchain Bootstrap
#
# PURPOSE: Configure complete native C++ build environment for DirectX 12
#
# WHAT IT DOES:
# 1. Detects/verifies CMake installation
# 2. Detects/verifies MSBuild (Visual Studio Build Tools)
# 3. Detects/installs/configures Ninja
# 4. Configures PATH environment variables
# 5. Outputs diagnostic report
#
# USAGE:
#   powershell -ExecutionPolicy Bypass -File bootstrap_toolchain.ps1
#
# RESULT:
#   All native build tools ready, PATH configured for current session
#   (Add to $PROFILE for permanent configuration)
#############################################################################

param(
    [switch]$Permanent = $false,
    [switch]$InstallNinja = $true,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Continue"

# Colors for output
$colors = @{
    Reset = "`e[0m"
    Green = "`e[92m"
    Yellow = "`e[93m"
    Red = "`e[91m"
    Cyan = "`e[96m"
    Bold = "`e[1m"
}

function Write-Status($msg, $color = "Cyan") {
    Write-Host "[$($colors[$color])*$($colors.Reset)] $msg"
}

function Write-Success($msg) {
    Write-Host "[$($colors.Green)OK$($colors.Reset)] $msg"
}

function Write-Error_($msg) {
    Write-Host "[$($colors.Red)FAIL$($colors.Reset)] $msg"
}

function Write-Warn($msg) {
    Write-Host "[$($colors.Yellow)WARN$($colors.Reset)] $msg"
}

# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

function Test-CMake {
    try {
        $ver = (cmake --version 2>&1 | Select-Object -First 1)
        if ($ver -match "cmake version") {
            Write-Success "CMake: $ver"
            return $true
        }
    } catch { }
    Write-Error_ "CMake not found or not in PATH"
    return $false
}

function Test-MSBuild {
    try {
        $ver = (msbuild -version 2>&1 | Select-Object -First 1)
        if ($ver -match "MSBuild") {
            Write-Success "MSBuild: $ver"
            return $true
        }
    } catch { }

    # Try to find MSBuild in known locations
    $msbuildPaths = @(
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
    )

    foreach ($path in $msbuildPaths) {
        if (Test-Path $path) {
            Write-Status "Found MSBuild at: $path"
            $script:MSBuildPath = $path
            return $true
        }
    }

    Write-Error_ "MSBuild not found in PATH or standard locations"
    return $false
}

function Test-Ninja {
    try {
        $ver = (ninja --version 2>&1)
        if ($ver) {
            Write-Success "Ninja: $ver"
            return $true
        }
    } catch { }

    # Check common installation paths
    $ninjaPaths = @(
        "C:\Program Files\Ninja\ninja.exe",
        "C:\Program Files (x86)\Ninja\ninja.exe",
        "$env:LOCALAPPDATA\Programs\Ninja\ninja.exe"
    )

    foreach ($path in $ninjaPaths) {
        if (Test-Path $path) {
            Write-Status "Found Ninja at: $path"
            $script:NinjaPath = $path
            return $true
        }
    }

    Write-Warn "Ninja not found in PATH or standard locations"
    return $false
}

# ============================================================================
# INSTALLATION FUNCTIONS
# ============================================================================

function Install-Ninja {
    Write-Status "Installing Ninja via winget..."

    try {
        # Check if already installed
        $installed = winget list --name "Ninja" 2>&1 | Select-String "Ninja-build"
        if ($installed) {
            Write-Warn "Ninja appears to already be installed, skipping install"
            return $true
        }

        # Install fresh
        Write-Status "Running: winget install Ninja-build --force"
        winget install Ninja-build --force

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Ninja installation succeeded"
            Start-Sleep -Seconds 2  # Give winget time to settle
            return $true
        } else {
            Write-Error_ "Ninja installation failed (exit code: $LASTEXITCODE)"
            return $false
        }
    } catch {
        Write-Error_ "Exception during Ninja installation: $_"
        return $false
    }
}

function Add-ToPath($path) {
    if ($path -and (Test-Path $path)) {
        if ($env:PATH -notlike "*$path*") {
            $env:PATH = "$path;$env:PATH"
            Write-Success "Added to PATH: $path"
            return $true
        } else {
            Write-Status "Already in PATH: $path"
            return $true
        }
    }
    return $false
}

# ============================================================================
# MAIN BOOTSTRAP
# ============================================================================

Write-Host ""
Write-Host "$($colors.Bold)$($colors.Cyan)========== MX2LM Native Toolchain Bootstrap ==========$($colors.Reset)"
Write-Host ""

$toolsFound = 0
$toolsTotal = 3

# Check CMake
Write-Status "Checking CMake..."
if (Test-CMake) { $toolsFound++ }
Write-Host ""

# Check MSBuild
Write-Status "Checking MSBuild (Visual Studio Build Tools)..."
if (Test-MSBuild) { $toolsFound++ }
Write-Host ""

# Check Ninja
Write-Status "Checking Ninja..."
if (Test-Ninja) { $toolsFound++ }
else {
    if ($InstallNinja) {
        Write-Status "Attempting to install Ninja..."
        if (Install-Ninja) {
            if (Test-Ninja) { $toolsFound++ }
        }
    }
}
Write-Host ""

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

Write-Status "Configuring PATH..."

if ($script:MSBuildPath) {
    $msbuildDir = Split-Path $script:MSBuildPath
    Add-ToPath $msbuildDir
}

if ($script:NinjaPath) {
    $ninjaDir = Split-Path $script:NinjaPath
    Add-ToPath $ninjaDir
}

Write-Host ""

# ============================================================================
# DIAGNOSTIC REPORT
# ============================================================================

Write-Host "$($colors.Bold)========== Diagnostic Report ==========$($colors.Reset)"
Write-Host ""

Write-Host "Tools Status: $toolsFound/$toolsTotal found/installed"
Write-Host ""

# CMake
Write-Host "$($colors.Bold)CMake:$($colors.Reset)"
try {
    $cmakePath = (Get-Command cmake -ErrorAction SilentlyContinue).Path
    Write-Host "  Path: $cmakePath"
    Write-Host "  Version: $(cmake --version | Select-Object -First 1)"
} catch {
    Write-Host "  $($colors.Red)Not available$($colors.Reset)"
}
Write-Host ""

# MSBuild
Write-Host "$($colors.Bold)MSBuild:$($colors.Reset)"
try {
    $msbuildPath = (Get-Command msbuild -ErrorAction SilentlyContinue).Path
    if (-not $msbuildPath -and $script:MSBuildPath) {
        $msbuildPath = $script:MSBuildPath
    }
    Write-Host "  Path: $msbuildPath"
    Write-Host "  Version: $(msbuild -version | Select-Object -First 1)"
} catch {
    Write-Host "  $($colors.Red)Not available$($colors.Reset)"
}
Write-Host ""

# Ninja
Write-Host "$($colors.Bold)Ninja:$($colors.Reset)"
try {
    $ninjaPath = (Get-Command ninja -ErrorAction SilentlyContinue).Path
    if (-not $ninjaPath -and $script:NinjaPath) {
        $ninjaPath = $script:NinjaPath
    }
    Write-Host "  Path: $ninjaPath"
    Write-Host "  Version: $(ninja --version)"
} catch {
    Write-Host "  $($colors.Red)Not available$($colors.Reset)"
}
Write-Host ""

# ============================================================================
# SUMMARY
# ============================================================================

Write-Host "$($colors.Bold)========== Summary ==========$($colors.Reset)"
Write-Host ""

if ($toolsFound -eq $toolsTotal) {
    Write-Success "All native build tools configured!"
    Write-Host ""
    Write-Host "You can now build DirectX 12 components:"
    Write-Host ""
    Write-Host "  cd C:\public_html\MX2LM\native\d3d12_compute"
    Write-Host "  .\build_sxme.bat"
    Write-Host ""
    exit 0
} else {
    Write-Error_ "Some tools are missing:"
    Write-Host ""

    if (-not (Test-CMake)) {
        Write-Host "  - Install CMake: winget install CMake"
    }
    if (-not (Test-MSBuild)) {
        Write-Host "  - Install Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/"
    }
    if (-not (Test-Ninja)) {
        Write-Host "  - Install Ninja: winget install Ninja-build"
    }

    Write-Host ""
    exit 1
}
