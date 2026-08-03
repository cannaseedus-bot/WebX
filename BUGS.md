# BUGS.md — Training Error Story

What the hardware told us. These bugs were found during real training runs.
Error logs are kept as evidence. Each bug has a design implication for future GPU code.

---

## Bug 1: D3D11 Device Removal During Adam Step (CRITICAL)

**Source:** `C:\Users\canna\.gpu_trainer\run_diag.txt`

**What happened:**
```
[trainer] init complete — 317 params, n_layer=24, n_embd=1024, n_head=16
[main] starting training: 2 steps
[dbg] h[0](embed): max=2.48785 ok
[dbg] h[1](l0 out): max=98.5713 ok
[dbg] h[NL](final): max=666.23 ok
[dbg] logits: max=341.276 ok
[dbg] loss=0.00552368 target=0
[adam] CreateQuery failed step=1 cqhr=0x887a0005 device_removed=0x887a0020
[trainer] step=1 loss=0.00552368
[dbg] h[0](embed): max=0 ok        ← all zeros: device was lost
[adam] device already lost BEFORE Adam step=2 reason=0x887a0020
[trainer] syncWeightsToCPU: synced=0 skipped=317   ← no weights actually saved
```

**Diagnosis:** Forward pass completed clean (loss=0.00552). The Intel HD 4600 driver
evicted the D3D11 device context during `ID3D11Query` creation at the start of Adam step 1.
DXGI_ERROR_DEVICE_REMOVED (0x887a0020) — iGPU shared-memory timeout or TDR reset.
Step 2 ran with a dead device: all activations = 0, model corrupted.

**Fix applied:** CPU Adam fallback (`run3_log.txt`):
```
[trainer] adam shader not found — using CPU Adam (OK)
[trainer] step=1 loss=0.00545017
[trainer] step=2 loss=0.000312377
...
[trainer] step=5 loss=2.94451e-05
```
Loss dropped cleanly. CPU Adam with D3D11 forward pass works fine.

**Design constraint:**
- D3D11 GPU Adam on shared-memory iGPU MUST have device-lost recovery
- Any GPU resource creation inside a training step must check for DXGI_ERROR_DEVICE_REMOVED
- Always implement CPU Adam fallback; never assume GPU Adam is available on iGPU

---

## Bug 2: XJSL Integration Broke createUAV Interface

**Source:** `v0.1.1-igpu-trainer-xjsl/build-local/` (first build = FAILED, second = fixed)

**What happened:**
```
gpt2_trainer.cpp(384,22): error C3861: 'createUAV': identifier not found
gpt2_trainer.cpp(385,22): error C3861: 'createUAV': identifier not found
gpt2_trainer.cpp(386,22): error C3861: 'createUAV': identifier not found
gpt2_trainer.cpp(387,22): error C3861: 'createUAV': identifier not found
gpt2_trainer.cpp(389,36): error C2466: cannot allocate an array of constant size 0
Build FAILED — 5 errors
```

**Diagnosis:** When XJSL was integrated into igpu-trainer, the D3D11 engine headers were
restructured. `createUAV()` was removed or renamed in the new XJSL-based engine interface.
`gpt2_trainer.cpp` still called the old API. The zero-size array error (C2466) followed
because the preceding UAV creation had failed.

**Fix applied:** Reconciled UAV API surface in `build-local/` (the successful build).
The `build/Release/build_output.log` shows a clean compile with XJSL assets copied.

**Design constraint:**
- GPU buffer creation API (createUAV, createSRV, createCBV) must be stable across shader DSL versions
- Abstract UAV creation behind an engine interface; never let trainer code call D3D11 types directly
- When integrating a new shader language (XJSL, KLSL, WGSL), run the full trainer build as a smoke test

---

## Bug 3: XShard Build — Space-in-Path CL.exe Failure

**Source:** `C:\Users\canna\.gpu_trainer\build_xshard_err.txt`

**What happened:**
```
cl : Command line warning D9024: unrecognized source file type 'Files'
cl : Command line warning D9024: unrecognized source file type '(x86)\Microsoft'
cl : Command line warning D9024: unrecognized source file type 'Visual'
...
xshard_validate.cpp
C:\Users\canna\.gpu_trainer\include\xshard.h(13): fatal error C1083:
    Cannot open include file: 'cstdint': No such file or directory
```

**Diagnosis:** The build script passed unquoted include paths to `cl.exe`:
```
cl /I C:\Program Files (x86)\Microsoft Visual Studio\...
```
CL.exe parsed each word after `/I` as a separate argument. `Files`, `(x86)\Microsoft`,
`Visual` became garbage filenames. The actual MSVC include path was never registered,
so `#include <cstdint>` could not resolve.

**Fix:** Quote all paths with spaces:
```
cl /I "C:\Program Files (x86)\Microsoft Visual Studio\..."
```
CMake handles this automatically. Direct CL.exe invocations do not.

**Design constraint:**
- All native build scripts that invoke `cl.exe` directly must quote every path argument
- Prefer CMake over hand-written build scripts for native code
- CI should build on a path with spaces (e.g., `C:\Program Files\`) to catch this class of bug

---

## Bug 4: Supernaut MicronauntFactory Hardcoded Dead Path

**Source:** `v3.2.0-supernaut/micronaut_factory.log`

**What happened:**
```
[INFO ] === Micronaut Factory v1.0.0 ===
[INFO ] Initializing MicronauntFactory at: C:\public_html\MX2LM\codex\AS-XCFE\micronaut
[INFO ] Registry does not exist, starting fresh
[INFO ] Scanning micronauts folder...
[WARN ] Micronauts directory does not exist: C:\public_html\MX2LM\codex\AS-XCFE\micronaut
[INFO ] Shutting down MicronauntFactory
[INFO ] === Complete ===
```
(Repeated 9+ times in under 1 second — tight retry loop with no backoff)

**Diagnosis:** MicronauntFactory has a hardcoded absolute path from a different machine layout.
When the path doesn't exist, the factory shuts down and is immediately re-spawned, creating
a spin loop. No configurable root, no exponential backoff.

**Fix needed:** `micronaut_factory.toml` with configurable root path. Backoff on failed init.

**Design constraint:**
- Factory base paths must be config-file driven (TOML/JSON), never hardcoded
- Any factory init that fails must use exponential backoff before retry
- The factory must log the config file path it's reading from, not just the resolved directory

---

## Bug 5: Pi-KUHUL Log UTF-8 Encoding Gap

**Source:** `C:\Users\canna\.gpu_trainer\pi_train.log`

**What happened:**
```
[pi-kuhul] each batch is routed across all 6 shards simultaneously
           data is NOT split ? every shard trains on every sample
```
The `?` should be `→` (U+2192, Unicode RIGHT ARROW).

**Diagnosis:** The Python training script printed UTF-8 to stdout, but the log capture pipeline
(PowerShell `Tee-Object` or cmd redirect) was in a non-UTF-8 codepage. The multi-byte UTF-8
sequence for `→` was mangled to `?`.

**Fix:**
```python
# At top of training script:
import sys
sys.stdout.reconfigure(encoding='utf-8')
```
Or set environment variable before launch: `$env:PYTHONIOENCODING = 'utf-8'`

**Design constraint:**
- All Python training scripts must explicitly set stdout to UTF-8
- KuhulCLI.bat must set `PYTHONIOENCODING=utf-8` before launching any Python process
- Log viewers must be configured for UTF-8 (Windows Terminal handles this; cmd.exe does not)

---

## Bug 6: GPT2Trainer D3D11 — Three Fixed NaN Bugs

**Source:** `C:\Users\canna\.gpu_trainer\TRAINER.md`

These bugs caused NaN loss values during D3D11 iGPU training. All three were fixed.

### 6a: SRV/UAV Aliasing in Final LayerNorm Backward
**Problem:** The same buffer was bound as both SRV (read) and UAV (write) in the LN backward shader.
D3D11 aliasing caused undefined behavior → NaN gradients.
**Fix:** Copy the LN output to a separate staging buffer before the backward pass.

### 6b: SRV/UAV Aliasing in GELU Backward
**Problem:** Same aliasing pattern in the GELU backward path. In-place gradient multiply
on a buffer that was still bound as SRV for the forward activations.
**Fix:** In-place gradient multiply workaround — write result to same buffer after unbinding SRV.

### 6c: Race Condition in LayerNorm dgamma/dbeta Accumulation
**Problem:** A single compute shader accumulated dgamma and dbeta across all sequence positions
in one pass. Multiple thread groups wrote to the same accumulation slot without proper ordering.
**Fix:** Split into two entry points — (1) per-position partial sums, (2) final per-dim reduction.

**Design constraint:**
- D3D11 does not support simultaneous SRV+UAV on the same subresource — enforce this at engine level
- Any reduction shader that accumulates across thread groups needs a two-pass design
- Add an assert/validation layer that checks for SRV/UAV aliasing before every dispatch

---

## Summary Table

| # | Bug | Severity | Fixed | Constraint |
|---|-----|----------|-------|-----------|
| 1 | D3D11 GPU Adam device removal | CRITICAL | CPU Adam fallback | Always implement CPU fallback |
| 2 | XJSL broke createUAV | HIGH | API reconciled | Abstract GPU buffer creation |
| 3 | XShard unquoted CL.exe paths | HIGH | Quote paths | All build scripts must quote paths |
| 4 | Factory hardcoded dead path | MEDIUM | Not yet fixed | Configurable root in TOML |
| 5 | UTF-8 log encoding | LOW | PYTHONIOENCODING | Set stdout UTF-8 explicitly |
| 6a | LN bwd SRV/UAV alias | HIGH | Staging buffer copy | No SRV+UAV aliasing in D3D11 |
| 6b | GELU bwd SRV/UAV alias | HIGH | In-place workaround | Same as 6a |
| 6c | LN dgamma/dbeta race | HIGH | Two-pass design | Reductions need two-pass |
