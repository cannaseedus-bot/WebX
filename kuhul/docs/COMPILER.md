# KUHUL Compiler Architecture

## Pipeline

```
Source text
    │
    ▼
KuhulLexer          (compiler/lexer.js)
    │  Token stream
    ▼
KuhulParser         (compiler/parser.js)
    │  AST
    ▼
SemanticAnalyzer    (compiler/semantic-analyzer.js)
    │  Annotated AST
    ▼
IRGenerator         (compiler/ir-generator.js)
    │  GeometricIR
    ▼
IROptimizer         (ir/ir-optimizer.js)
    │  Optimised GeometricIR
    ▼
IRVerifier          (ir/ir-verifier.js)
    │  Validated GeometricIR
    ▼
Codegen             (compiler/codegen/{js,wasm,webgpu}-codegen.js)
    │  Target source
    ▼
ExecutableProgram
```

## Modules

| Module | Responsibility |
|--------|---------------|
| `lexer.js` | Tokenise source text |
| `parser.js` | Build AST from tokens |
| `semantic-analyzer.js` | Type-check and validate |
| `ir-generator.js` | Lower AST to GeometricIR |
| `ir/ir-optimizer.js` | Optimise IR (dead-store, const-fold) |
| `ir/ir-verifier.js` | Validate IR before codegen |
| `codegen/js-codegen.js` | Emit JavaScript |
| `codegen/wasm-codegen.js` | Emit WAT (WebAssembly Text) |
| `codegen/webgpu-codegen.js` | Emit WGSL (WebGPU) |

## Adding a new target

1. Create `compiler/codegen/my-codegen.js` extending `BaseCodegen`.
2. Implement `generate(ir)` returning a target source string.
3. Register it in `kuhul-compiler.js` `CODEGENS` map.
