# WebX Build Guide

## Prerequisites

- Windows x64
- Visual Studio 2022 with Desktop development with C++
- CMake 3.20 or newer
- Windows 10 SDK

The canonical native host is `kuhul_engine.exe`. `MicrosoftSDK.ps1` is the
model-agnostic orchestration and task-list boundary; it does not replace the
native execution host.

## Recommended Release Build

From the repository root:

```powershell
.\build_release.bat
```

The build produces:

```text
build\bin\Release\kuhul_engine.exe
build\bin\Release\kuhul_opencl_helper.exe
build\bin\Release\kuhul_task_helper.exe
```

The helper executables are required for isolated OpenCL probing and
allowlisted TaskEngine actions.

## Manual CMake Build

Run from an x64 Native Tools PowerShell or Developer Command Prompt:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel
```

Build individual targets when iterating:

```powershell
cmake --build build --config Release --target kuhul_engine
cmake --build build --config Release --target kuhul_opencl_helper
cmake --build build --config Release --target kuhul_task_helper
```

## Diagnostics

```powershell
& ".\build\bin\Release\kuhul_engine.exe" --version
& ".\build\bin\Release\kuhul_engine.exe" --providers
& ".\build\bin\Release\kuhul_engine.exe" --test all
```

PowerShell requires `&` when invoking a quoted executable path.

## OpenCL Provider Setup

Set the driver directory before running provider or task diagnostics:

```powershell
$env:KUHUL_DRIVER_ROOT = "C:\DRIVERS\VDO\h2vdo66us14\Gfx\950fd7b1-7601-4c2a-ae29-d33825f748c9"
& ".\build\bin\Release\kuhul_engine.exe" --providers
```

The native host delegates platform/device enumeration to
`kuhul_opencl_helper.exe`; the helper is bounded by the host timeout.

## TaskEngine

Plan a task list without executing provider actions:

```powershell
& ".\build\bin\Release\kuhul_engine.exe" task-engine native\runtime\TaskList.json
```

Run only explicitly allowlisted helper actions:

```powershell
& ".\build\bin\Release\kuhul_engine.exe" task-run native\runtime\TaskList.json
```

`probe_provider`, `probe_opencl`, `stream_xshard`, and `verify_scx` are
supported explicit helper actions. Set `KUHUL_XSHARD_ROOT` before running the
sample GPT-OSS task list:

```powershell
$env:KUHUL_XSHARD_ROOT = "E:\models\GPT-OSS\hf"
& ".\build\bin\Release\kuhul_engine.exe" task-run native\runtime\TaskList.json
```

`stream_xshard` is bounded to four tiles. `verify_scx` uses the D3D11 feature-level 11_1 / `cs_5_0` smoke path by
default. It uses WARP for the correctness baseline; set
`KUHUL_D3D11_HARDWARE=1` to opt into hardware execution. The current Intel
hardware path may return `0xC0000005`, while WARP provides a stable bounded
validation path.

## MicrosoftSDK Orchestration

Inspect the Semantic Kernel capability manifest:

```powershell
& ".\native\semantic-kernel\MicrosoftSDK.ps1" -Command manifest
```

Generate a declarative task list:

```powershell
& ".\native\semantic-kernel\MicrosoftSDK.ps1" `
  -Command tasklist `
  -Prompt "Describe the requested build or runtime task"
```

The generated task list must be validated and admitted by `kuhul_engine.exe`;
MicrosoftSDK does not execute arbitrary tools or generated handlers.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `cmake` is not recognized | Use `C:\Program Files\CMake\bin\cmake.exe` or add CMake to `PATH`. |
| `cl.exe` is not recognized | Open an x64 Native Tools prompt or run `vcvars64.bat`. |
| OpenCL helper timeout | The driver enumeration is isolated; inspect the provider DLLs and `KUHUL_DRIVER_ROOT`. |
| Provider DLL not found | Confirm the DLL exists under `KUHUL_DRIVER_ROOT` and matches x64. |
| Task is blocked | Check provider admission and whether the action has an explicit helper contract. |
