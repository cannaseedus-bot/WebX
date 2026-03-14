# K'UHUL++ API Reference

## ExecutionEngine

The top-level pipeline: source → tokens → AST → IR → execution.

```typescript
import { ExecutionEngine } from './kuhul/runtime/execution-engine.js';

const engine = new ExecutionEngine({ strictMode: false });

// Run source end-to-end
const result = await engine.run(source, env);
// result: { output: Map, phase: number, log: string[], durationMs: number }

// Compile only (no execution)
const ir = engine.compile(source);
```

## tokenize

```typescript
import { tokenize } from './kuhul/compiler/lexer.js';
const tokens = tokenize(source); // Token[]
```

## parse

```typescript
import { parse } from './kuhul/compiler/parser.js';
const ast = parse(tokens); // ProgramNode
```

## analyze

```typescript
import { analyze } from './kuhul/compiler/semantic-analyzer.js';
const { errors, warnings, ast } = analyze(programNode);
```

## generateIR

```typescript
import { generateIR } from './kuhul/compiler/ir-generator.js';
const ir = generateIR(annotatedAST); // GeometricIR
```

## IRBuilder

```typescript
import { IRBuilder } from './kuhul/ir/ir-builder.js';

const ir = new IRBuilder()
    .setManifold(3, 'euclidean')
    .beginPhase('main', 0, 2 * Math.PI)
    .addConst('a', 1.0, { kind: 'scalar', dtype: 'float32' })
    .addConst('b', 2.0, { kind: 'scalar', dtype: 'float32' })
    .addGlyphOp('r', '⊕', 'a', 'b', { kind: 'scalar', dtype: 'float32' })
    .endPhase()
    .build();
```

## IROptimizer

```typescript
import { IROptimizer } from './kuhul/ir/ir-optimizer.js';
const optimIR = new IROptimizer().optimize(ir);
```

## IRVerifier

```typescript
import { IRVerifier } from './kuhul/ir/ir-verifier.js';
const { valid, errors, warnings } = new IRVerifier().verify(ir);
```

## KuhulVM

```typescript
import { KuhulVM } from './kuhul/runtime/kuhul-vm.js';
const result = await new KuhulVM().execute(ir, env);
```

## Code Generators

```typescript
import { JSCodegen }      from './kuhul/compiler/codegen/js-codegen.js';
import { WasmCodegen }    from './kuhul/compiler/codegen/wasm-codegen.js';
import { WebGPUCodegen }  from './kuhul/compiler/codegen/webgpu-codegen.js';

const js   = new JSCodegen().generate(ir);      // string (ES module)
const wasm = new WasmCodegen().generate(ir);    // Uint8Array
const wgsl = new WebGPUCodegen().generate(ir);  // string (WGSL)
```

## Standard Library

```typescript
// Glyph implementations
import { vectorEncrypt }       from './kuhul/stdlib/glyphs/vector-encrypt.js';
import { rotationalCompression}from './kuhul/stdlib/glyphs/compression.js';
import { sphericalLoop }       from './kuhul/stdlib/glyphs/spherical-loop.js';
import { torsionField }        from './kuhul/stdlib/glyphs/torsion-field.js';
import { radialProjection }    from './kuhul/stdlib/glyphs/radial-projection.js';

// Math
import { mat4Multiply, vec3Cross } from './kuhul/stdlib/math/linear-algebra.js';
import { tensorProduct }           from './kuhul/stdlib/math/tensor-ops.js';
import { distance3D, computeAABB } from './kuhul/stdlib/math/geometry.js';

// I/O
import { readKuhulFile }  from './kuhul/stdlib/io/file-ops.js';
import { uploadToGPU }    from './kuhul/stdlib/io/gpu-transfer.js';
```

## Tools

```typescript
import { KuhulREPL }     from './kuhul/tools/repl.js';
import { KuhulDebugger } from './kuhul/tools/debugger.js';
import { KuhulProfiler } from './kuhul/tools/profiler.js';
import { KuhulLinter }   from './kuhul/tools/linter.js';
```

---

*See also: [KUHUL.md](KUHUL.md), [COMPILER.md](COMPILER.md), [EXAMPLES.md](EXAMPLES.md)*
