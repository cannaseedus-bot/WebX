# Native Toolchain Bootstrap Guide

**Purpose**: Configure complete C++ build environment for DirectX 12 in one command

**Status**:
- ✅ MSBuild (Dev18) - INSTALLED
- ✅ CMake - INSTALLED
- ⏳ Ninja - NEEDS INSTALLATION/PATH CONFIG

---

## Quick Start (1 command)

```powershell
cd C:\public_html\MX2LM\native
powershell -ExecutionPolicy Bypass -File bootstrap_toolchain.ps1
```

**This will**:
1. ✅ Verify CMake is installed
2. ✅ Verify MSBuild is installed (Dev18)
3. ⏳ Install Ninja (if needed)
4. ✅ Add all tools to current session PATH
5. 📋 Print diagnostic report

---

## What Gets Configured

### MSBuild (Already Installed ✅)
```
Found: C:\Program Files\Microsoft Visual Studio\2022\BuildTools
Version: 18.3.0-release-26070-10+3972042b7
Status: READY
```

### CMake (Verify Installed)
```
Command: cmake --version
Required for: Project configuration
```

### Ninja (Install + Configure)
```
Install via: winget install Ninja-build
Configure: Add to PATH
```

---

## Usage Examples

### Example 1: Full Bootstrap (Recommended)

```powershell
cd C:\public_html\MX2LM\native
powershell -ExecutionPolicy Bypass -File bootstrap_toolchain.ps1
```

**Output**:
```
[*] Checking CMake...
[OK] CMake: cmake version 3.28.1
[*] Checking MSBuild...
[OK] MSBuild: MSBuild version 18.3.0
[*] Checking Ninja...
[WARN] Ninja not found, attempting install...
[*] Running: winget install Ninja-build --force
[OK] Ninja installation succeeded
[OK] Ninja: 1.11.1

Tools Status: 3/3 found/installed
```

Then build DirectX:
```powershell
cd C:\public_html\MX2LM\native\d3d12_compute
.\build_sxme.bat
```

### Example 2: Check Only (Don't Install)

```powershell
powershell -ExecutionPolicy Bypass -File bootstrap_toolchain.ps1 -InstallNinja:$false
```

### Example 3: Verbose Output

```powershell
powershell -ExecutionPolicy Bypass -File bootstrap_toolchain.ps1 -Verbose
```

---

## Permanent PATH Configuration

The bootstrap script configures PATH for the **current PowerShell session only**.

To make it permanent, add to your PowerShell profile:

```powershell
# Edit profile
notepad $PROFILE

# Add these lines:
# Ninja
$env:PATH = "C:\Program Files\Ninja;$env:PATH"

# MSBuild (if not already in PATH)
$env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin;$env:PATH"
```

Then reload:
```powershell
. $PROFILE
```

---

## Troubleshooting

### "CMake not found"
```powershell
winget install CMake
```

### "MSBuild not found"
Visual Studio Build Tools is already installed but may not be in PATH:
```powershell
# Verify installation
Test-Path "C:\Program Files\Microsoft Visual Studio\2022\BuildTools"

# If exists, add to PATH (bootstrap script does this)
# Or install fresh: Visual Studio Installer → Modify
```

### "Ninja installation failed"
```powershell
# Manual install
winget install Ninja-build --force

# Or download: https://github.com/ninja-build/ninja/releases
# Extract to C:\Program Files\Ninja
# Add to PATH manually
```

### "After Ninja install, still not found"
Restart PowerShell:
```powershell
exit
# Reopen PowerShell
powershell -ExecutionPolicy Bypass -File bootstrap_toolchain.ps1
```

---

## Full Toolchain Verification

After bootstrap succeeds, verify everything works:

```powershell
# All should return version numbers
cmake --version
msbuild -version
ninja --version

# Then build DirectX
cd C:\public_html\MX2LM\native\d3d12_compute
.\build_sxme.bat

# Test
build\Release\d3d12_compute_demo.exe
# Expected: D3D12 result: 1024, 1024, 1024, 1024
```

---

## Architecture

```
bootstrap_toolchain.ps1
├─ Test-CMake
├─ Test-MSBuild
│  └─ Searches standard VS Build Tools location
├─ Test-Ninja
│  └─ Searches multiple install paths
├─ Install-Ninja (if not found)
│  └─ winget install Ninja-build --force
├─ Add-ToPath (all tools)
│  └─ Configures $env:PATH for current session
└─ Diagnostic Report
   └─ Verifies all tools and shows their versions
```

---

## Next Steps

After bootstrap succeeds:

```powershell
# 1. Build DirectX 12 compute shader
cd C:\public_html\MX2LM\native\d3d12_compute
.\build_sxme.bat

# 2. Test GPU demo
build\Release\d3d12_compute_demo.exe

# 3. Train with GPU acceleration
cd ..\..\models\SCX-MoE\trainer
python train_real_full.py --epochs 1 --use-directx
```

---

## Script Options

```powershell
# Default (install Ninja, configure PATH)
./bootstrap_toolchain.ps1

# Don't install Ninja (check only)
./bootstrap_toolchain.ps1 -InstallNinja:$false

# Verbose output
./bootstrap_toolchain.ps1 -Verbose

# Make PATH changes permanent (future enhancement)
./bootstrap_toolchain.ps1 -Permanent
```

---

**Status**: Ready to configure complete native build toolchain ✅
