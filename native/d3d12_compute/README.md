# D3D12 Compute Demo (Win10)

Minimal Direct3D 12 compute sample that includes:

- Vector add (baseline compute sanity check).
- Triangle-mesh propagation kernel (mirrors the GSNR browser runtime).
- KUHUL glyph adapter (run the same glyph program natively).
- Timing + fence metrics (CPU wall-clock + fence value).

## Build (Developer Command Prompt)

```
cd native\d3d12_compute
build.bat
```

## Run

```
build\Release\d3d12_compute_demo.exe
```

Example output (first values):

```
D3D12 result: 1024, 1024, 1024, 1024
Triangle mesh output: 0.92, 0.92, 0.92, 0.92
Triangle steps: 8 in 0.45 ms | steps/sec=17777.8 | tris/sec=568889 | fence=6
KUHUL backend: D3D12
KUHUL vector add: 1024, 1024, 1024, 1024
KUHUL triangle step: 0.92, 0.92, 0.92, 0.92
```

## What’s Included

- `shader.hlsl`: Vector add compute shader (`cs_5_0`).
- `triangle.hlsl`: Triangle propagation kernel with neighbor coupling + phase update.
- `main.cpp`: Native D3D12 runtime, ping‑pong buffers, and KUHUL glyph adapter.

## Notes

- Uses `D3DCompileFromFile` and `cs_5_0` for broad compatibility.
- Requires the Windows 10 SDK (D3D12 + D3DCompiler) and Visual Studio Build Tools.
- Triangle propagation uses a simple ring neighbor list (2 neighbors) and runs 8 steps by default.
- KUHUL adapter supports:
  - `Wo` (allocate buffer)
  - `Sek.vector_add`
- `Sek.triangle_step`
- Timing is measured around command submission + fence completion.

## SCX-MoE Runtime Adapter (Hardware/WARP/CPU)

`sxme_compute.dll` now probes runtime adapters in this order:
1. D3D12 hardware adapter
2. D3D12 WARP (software rasterizer)
3. CPU fallback in Python (`directx_wrapper.py`)

For older iGPUs (for example Intel HD 4600 with FL 11.1 drivers), D3D12 hardware
device creation may fail and the runtime may land on WARP. By default, Python
wrapper treats WARP as disabled and falls back to CPU unless explicitly enabled:

```powershell
set SCXMOE_ALLOW_WARP=1
python train_real_full.py --epochs 1 --use-directx
```

Native exports for diagnostics:
- `SCXMoEProbeRuntime()`
- `SCXMoEGetRuntimeAdapterInfo(char* out, size_t outSize, int* backendCode)`
  - backend codes: `0=none, 1=hardware, 2=warp`

## GSNR Dispatch

This runtime can load `mesh.xml` or `meshx.bin` from a `model.gsnr` directory:

```
d3d12_compute_demo.exe --gsnr path\to\model.gsnr
```

Optional steps:

```
d3d12_compute_demo.exe --gsnr path\to\model.gsnr --steps 16
```

When `--gsnr` is used the runtime:

- Loads `meshx.bin` if present (fast binary streaming).
- Otherwise parses `mesh.xml` vertices/triangles.
- Uses precomputed neighbors if `n0/n1/n2` are present on `<triangle>` tags.
- Falls back to CPU adjacency build if neighbors are missing.
- Streams vertex/triangle/neighbor buffers to GPU.
- Loads `tensors_fp32.tbin` (if present) and binds it as a structured buffer.

## DDS State Initialization

If `phase.dds` or `output.dds` exist in the GSNR directory they are used to initialize buffers.

Supported DDS format:
- DX10 `R32_FLOAT` (width*height floats)

## MeshX Writer

Use the helper script to generate `meshx.bin`:

```
python scripts/meshx_from_xml.py --xml model.gsnr\mesh.xml --out model.gsnr\meshx.bin
```

## Tensor Bin (FP32 baseline)

If `tensors_fp32.tbin` exists in the GSNR directory, the runtime loads it and binds
it as `TensorWeights` (t4). The triangle kernel uses `TensorWeights[0]` as a baseline
weight multiplier.

Generate it via K‑Shell:

```
build\kshell.exe
> tensors compile
```

This is intended as a native GPU path when browser WebGPU is blocked.
