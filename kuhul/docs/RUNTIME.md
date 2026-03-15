# KUHUL Runtime Guide

## Architecture

```
KuhulVM
  ├─ MemoryManager   (manifold M)
  ├─ PhaseManager    (phase cycles)
  └─ ExecutionEngine (backend selection)
       ├─ CPU (default)
       ├─ WebGPU (if available)
       └─ WebAssembly (if available)
```

## KuhulVM

Executes a `GeometricIR` program on an in-process virtual machine.

```js
import { KuhulVM } from './runtime/kuhul-vm.js';
const vm = new KuhulVM();
const state = await vm.execute(ir, { /* initial context */ });
```

Returns a plain object mapping identifier names to `Float32Array` contents.

## MemoryManager

Manages typed Float32 memory blocks identified by integer IDs.

```js
import { MemoryManager } from './runtime/memory-manager.js';
const mm = new MemoryManager();
const id = mm.allocate(1024);   // 1024 float32 elements
const blk = mm.get(id);         // { id, buffer: Float32Array, size }
mm.free(id);
```

## PhaseManager

Advances a phase angle from 0 to 2π in discrete steps.

```js
import { PhaseManager } from './runtime/phase-manager.js';
const pm = new PhaseManager(Math.PI / 8);
pm.nextPhase();              // → 0.3927...
pm.getCurrentPhase();        // current angle
pm.reset();
```

## Scheduler

Runs async tasks in priority order.

```js
import { Scheduler } from './runtime/scheduler.js';
const s = new Scheduler();
s.schedule({ fn: async () => 'done', priority: 10, name: 'my-task' });
const results = await s.run();
```
