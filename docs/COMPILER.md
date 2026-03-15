# K'UHUL++ Compiler Design

## Pipeline Overview

```
Source (.kuhul)
    │
    ▼
┌─────────────┐
│   Lexer     │  tokenize(source) → Token[]
│ lexer.ts    │  Handles Unicode glyphs, π literals, keywords
└──────┬──────┘
       │ Token[]
       ▼
┌─────────────┐
│   Parser    │  parse(tokens) → ProgramNode
│ parser.ts   │  Recursive-descent, produces AST
└──────┬──────┘
       │ ProgramNode
       ▼
┌─────────────────┐
│ Semantic Anal.  │  analyze(ast) → AnnotatedAST
│ semantic-      │  Symbol resolution, type inference
│ analyzer.ts    │  Glyph validation, π-range checks
└──────┬──────────┘
       │ AnnotatedAST
       ▼
┌──────────────┐
│ IR Generator │  generateIR(ast) → GeometricIR
│ ir-gen.ts    │  SSA-style IR, phase tracking
└──────┬───────┘
       │ GeometricIR
       ▼
┌──────────────┐
│ IR Optimizer │  optimize(ir) → GeometricIR
│ ir-optim.ts  │  Constant folding, DCE, transform fusion
└──────┬───────┘
       │ GeometricIR
       ▼
┌──────────────┐
│ IR Verifier  │  verify(ir) → ValidationResult
│ ir-verif.ts  │
└──────┬───────┘
       │ GeometricIR
    ┌──┴──────────────────┐
    │  Code Generators     │
    ├─────────────────────┤
    │ JSCodegen   → .js   │  kuhul/compiler/codegen/js-codegen.ts
    │ WasmCodegen → .wasm │  kuhul/compiler/codegen/wasm-codegen.ts
    │ WebGPUCodegen→ WGSL │  kuhul/compiler/codegen/webgpu-codegen.ts
    └─────────────────────┘
```

## Token Types

| Type       | Examples                        |
|------------|---------------------------------|
| `NUMBER`   | `42`, `3.14`                    |
| `STRING`   | `"hello"`, `'world'`            |
| `KEYWORD`  | `Tensor`, `Model`, `if`, `for`  |
| `GLYPH`    | `⊗`, `⊕`, `⤍`, `↻`             |
| `PI`       | `π`                             |
| `PI_EXPR`  | `0.5π`, `2π`                    |
| `EOF`      | End of source                   |

## AST Node Kinds

| Kind             | Description                     |
|------------------|---------------------------------|
| `Program`        | Root node                       |
| `TensorDecl`     | `Tensor x = expr`               |
| `ClusterDecl`    | `Cluster name { ... }`          |
| `ModelDecl`      | `Model name { ... }`            |
| `PipelineDecl`   | `Pipeline name { ... }`         |
| `GlyphOp`        | `left ⊗ right`                  |
| `Assignment`     | `target = value`                |
| `BinaryExpr`     | `a + b`, `a * b`                |
| `FunctionCall`   | `fn(args)`                      |
| `IfStmt`         | `if (cond) { ... }`             |
| `ForStmt`        | `for (init; test; update) { }` |
| `WhileStmt`      | `while (cond) { ... }`          |
| `PiExpr`         | `0.5π`                          |

## IR Instruction Ops

| Op           | Description                              |
|--------------|------------------------------------------|
| `const`      | Numeric or string constant               |
| `load`       | Load from symbol table                   |
| `store`      | Store to named location                  |
| `alloc`      | Allocate typed buffer                    |
| `invoke`     | Call a function or built-in              |
| `phase`      | Advance geometric phase by delta         |
| `branch`     | Unconditional branch                     |
| `condbranch` | Conditional branch                       |
| `label`      | Basic block label                        |
| `return`     | Return from function / program           |
| `⊗` `⊕` … | Glyph operation instructions             |

---

*See also: [KUHUL.md](KUHUL.md), [IR.md](IR.md)*
