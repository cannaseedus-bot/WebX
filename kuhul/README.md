# KUHUL – Tensor Language for Geometric Computation

`@webx/kuhul` is a complete language implementation for **KUHUL**, a
tensor-oriented language designed for GPU-accelerated geometric pipelines in
the D12WebX runtime.

---

## Features

- **Bracket-based syntax** – every statement is `[Keyword args…]`
- **Seven glyph operators** – ⊗ ⊕ ⊖ ⊛ ⊜ ⊝ ⊞
- **Three compilation targets** – JavaScript, WebAssembly (WAT), WebGPU (WGSL)
- **Full pipeline** – Lexer → Parser → Semantic → IR → Optimizer → Verifier → Codegen
- **LLM tooling** – completion, validation, optimisation, documentation
- **Developer tools** – REPL, debugger, profiler, linter, formatter

---

## Quick Start

```js
import { KuhulCompiler } from './kuhul/index.js';

const compiler = new KuhulCompiler();
const program  = await compiler.compile(`
[Pop]
  [Wo X tensor<float32, [10]>]
  [Yax X]
  [Ch'en X X]
[Xul]
`, 'js');

console.log(program.code);
```

---

## Language Overview

| Construct | Syntax |
|-----------|--------|
| Fold (block) | `[Pop] … [Xul]` |
| Allocate | `[Wo name tensor<type, [dims]>]` |
| Read | `[Yax name]` |
| Write | `[Ch'en name value]` |
| Operation | `[Sek ⊗ A B]` |
| Phase cycle | `[K'ayab'] … [Kumk'u]` |
| Invocation | `[Muwan fn arg…]` |

### Glyphs

| Symbol | Meaning |
|--------|---------|
| `⊗` | Tensor product / matmul |
| `⊕` | Addition |
| `⊖` | Subtraction |
| `⊛` | Convolution |
| `⊜` | Equality |
| `⊝` | Negation |
| `⊞` | Direct sum / concat |

---

## Directory Structure

```
kuhul/
├── compiler/           Lexer, parser, semantic, codegen
│   └── codegen/        JS, Wasm, WebGPU backends
├── grammar/            EBNF grammar + parser/validator
├── ir/                 GeometricIR types, builder, optimizer, verifier
├── llm/                LLM-assisted tools
├── runtime/            VM, phase manager, memory, engine, scheduler
├── stdlib/             Glyphs, math, tensor, I/O
├── tools/              REPL, debugger, profiler, linter, formatter
├── tests/              Jest test suite
├── examples/           .kuhul example programs
├── docs/               Documentation
└── index.js            Public API
```

---

## Running Tests

```bash
cd /path/to/WebX
npm test
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| [LANGUAGE.md](docs/LANGUAGE.md) | Language specification |
| [GRAMMAR.md](docs/GRAMMAR.md) | Formal EBNF grammar |
| [COMPILER.md](docs/COMPILER.md) | Compiler architecture |
| [IR.md](docs/IR.md) | IR specification |
| [RUNTIME.md](docs/RUNTIME.md) | Runtime guide |
| [LLM.md](docs/LLM.md) | LLM integration |
| [API.md](docs/API.md) | Full API reference |
| [EXAMPLES.md](docs/EXAMPLES.md) | Code examples |
