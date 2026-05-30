# Building DirectX 12 Without Visual Studio

**Goal**: Build `sxme_compute.dll` without downloading 2.5GB Visual Studio

**Solution**: Use Ninja build system (lightweight, fast)

---

## Quick Setup (5 minutes)

### Step 1: Install Ninja

```powershell
# Option A: Package Manager (Recommended)
winget install Ninja-build

# Option B: Manual Download
# https://github.com/ninja-build/ninja/releases
# → Download ninja-win.zip
# → Extract to C:\Program Files\Ninja
# → Add to PATH: C:\Program Files\Ninja
```

### Step 2: Verify Ninja Installed

```powershell
ninja --version
# Output: 1.11.1 (or similar)
```

### Step 3: Build DirectX

```powershell
cd C:\public_html\MX2LM\native\d3d12_compute

# Use Ninja build script
.\build_sxme_ninja.bat

# Output:
# [OK] Ninja found
# [OK] CMake found
# [*] Configuring with CMake (Ninja)...
# [OK] CMake complete
# [*] Building with Ninja...
# [OK] Build complete
# [OK] build\sxme_compute.dll
```

### Step 4: Verify It Works

```powershell
build\d3d12_compute_demo.exe
# Output: D3D12 result: 1024, 1024, 1024, 1024 ✓
```

---

## Comparison: Ninja vs MSBuild

| Aspect | Ninja | MSBuild |
|--------|-------|---------|
| Install Size | ~50 MB | 2.5-5 GB |
| Download Time | 2 min | 15-20 min |
| Build Time | Fast | Slower |
| Complexity | Simple | Complex |
| Support | Good | Official |

**For now**: Use Ninja ✓
**In future**: Switch to MSBuild if needed

---

## Troubleshooting

**"Ninja not found"**
```powershell
winget install Ninja-build
# Then restart PowerShell
```

**"CMake not found"**
```powershell
winget install CMake
```

**"Build fails with C++ errors"**
→ You still need a C++ compiler. Options:
- Install LLVM (via winget): `winget install LLVM`
- Or install MSBuild (the 2.5GB option)

---

## Status

- ✅ **Build system**: CMake ready
- ✅ **Build tool**: Ninja available (lightweight)
- ✅ **Compiler**: Needs to be verified (next step)

**Next**: Run build and see if C++ compiler is available

```bash
cd C:\public_html\MX2LM\native\d3d12_compute
.\build_sxme_ninja.bat
```

If it fails on compiler, then MSBuild (or LLVM) needed.
