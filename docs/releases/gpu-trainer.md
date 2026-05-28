# .gpu_trainer — Audit (XVM 1000-fiber + KLSL + Training History)

**Status:** Fully audited. Merge branches: `merge/xvm-1000-fiber`

## XVM 1000-Fiber CUDA Emulator
- Warp vote: OP_VOTE_ALL(0x30), OP_VOTE_ANY(0x31), OP_BALLOT(0x32)
- Warp shuffle: OP_SHUFFLE_XOR(0x33), OP_SHUFFLE_UP(0x34), OP_SHUFFLE_DOWN(0x35)
- Tensor core emulation: OP_WMMA_COMPUTE(0x42) — 16×16×16 matmul-accumulate
- Dynamic parallelism: OP_LAUNCH_KERNEL(0x50), OP_LAUNCH_GRID(0x51)
- Async streams: OP_STREAM_CREATE(0x60), OP_STREAM_LAUNCH(0x61), OP_STREAM_SYNC(0x62)
- ClusterConfig: totalCores=1000, cpuThreads=8, sharedMemoryKB=1024, ~50 MIPS
- EBPD DLL exports: EBPD_CreateVirtualCluster, EBPD_LaunchVirtualCores, EBPD_GetResult

## KLSL Compiler
- Glyph sigil language: ⟁ shader_name...⟁Xul⟁, ⟁Wo⟁/⟁Sek⟁/⟁Ch'en⟁/⟁Yax⟁/[Pop/Xul]/⟁K'ayab'⟁
- Targets: HLSL text, XVM bytecode, WGSL stub
- 50+ opcodes: control, int/float arithmetic, FMAD fused, trig, memory, atomic, thread

## XCFE JSON Program Format
- @ops[], @state{}, @control{}, @runtime{}, @buffers[]
- Op types: pure, primitive, gpu, agent — embeds KLSL glyph strings directly

## Training History
- Pi-KUHUL: token_pos % 6 → shard_id routing, balanced 6-shard simultaneous training
- Pi-phase tracking: counter +0.05/step, checkpoint every 500 steps
- Final state: step=27833, loss~0.0003, 52.95M params
- Throughput: 285 tok/s (fine-tune) vs 185 tok/s (pre-train)
- 4 LoRA adapters (rank-8, ~295K params each): commands/tools/micronauts/agents, 2000 steps each
- Export: .pt → pi_kuhul_adapter_to_scxq2.py → .scxq2 (7.8MB INT4 each)

## Key Shaders
- kuhul_fold_compute.hlsl: MM-1 matmul + top-1 routing, CM-1 gate check (ControlFlags[0]==0x0002)
- scxq2_int4_decode.hlsl: fused INT4 dequant + GEMM
- scxq2_infer_layer.hlsl: full inference pass
- pi_field.wgsl: WebGPU pi-field compute

## Sub-directories with logs/history (read during merge)
- trainer/pi_kuhul_train_hd4600.py — HD 4600-specific training script
- trainer/pi_kuhul_adapter_mutate.py — mutation engine (auto-retrain on usage events)
- 07_transformer/model/foundation/ — final_3way.pt + .scxq2
- 07_transformer/model/complete_model/ — fully merged model
- KUHUL_V1/MODELS.md — three model lineages (KUHUL/.xhard/tiny.x)
- KUHUL_V1/TODO.md — training queue + completed milestones

## Merge Targets
- src/xvm/fiber-1000.js
- src/klsl/
- src/xcfe/
- src/adapters/adapter-loader.js
- src/adapters/adapter-registry.js
- shaders/ (copy gpt2_*.hlsl + scxq2_*.hlsl + pi_field.wgsl)
