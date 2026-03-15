# KUHUL API Reference

## Compiler

### `KuhulCompiler`

```js
import { KuhulCompiler } from './compiler/kuhul-compiler.js';
const compiler = new KuhulCompiler();
const program  = await compiler.compile(source, target);
// target: 'js' | 'wasm' | 'webgpu'
```

### `KuhulLexer`

```js
import { KuhulLexer } from './compiler/lexer.js';
const tokens = new KuhulLexer().lex(source); // Token[]
```

### `KuhulParser`

```js
import { KuhulParser } from './compiler/parser.js';
const ast = new KuhulParser().parse(tokens);
```

### `SemanticAnalyzer`

```js
import { SemanticAnalyzer } from './compiler/semantic-analyzer.js';
const { errors, warnings } = new SemanticAnalyzer().analyze(ast);
```

### `IRGenerator`

```js
import { IRGenerator } from './compiler/ir-generator.js';
const ir = new IRGenerator().generate(ast);
```

---

## IR

### `IRBuilder`

```js
import { IRBuilder } from './ir/ir-builder.js';
const ir = new IRBuilder()
  .createProgram()
  .addInstruction('ALLOC', ['X', 'float32', [10]])
  .build();
```

### `IROptimizer`

```js
import { IROptimizer } from './ir/ir-optimizer.js';
const optimised = new IROptimizer().optimize(ir);
```

### `IRVerifier`

```js
import { IRVerifier } from './ir/ir-verifier.js';
const { valid, errors } = new IRVerifier().verify(ir);
```

### `IRPrinter`

```js
import { IRPrinter } from './ir/ir-printer.js';
console.log(new IRPrinter().print(ir));
```

---

## Runtime

### `KuhulVM`

```js
import { KuhulVM } from './runtime/kuhul-vm.js';
const state = await new KuhulVM().execute(ir, context);
```

### `PhaseManager`

```js
import { PhaseManager } from './runtime/phase-manager.js';
const pm = new PhaseManager();
pm.nextPhase(); pm.getCurrentPhase(); pm.reset();
```

### `MemoryManager`

```js
import { MemoryManager } from './runtime/memory-manager.js';
const mm = new MemoryManager();
const id = mm.allocate(1024);
mm.free(id);
```

---

## Stdlib

### `executeGlyph(glyph, a, b, opts)`

```js
import { executeGlyph } from './stdlib/glyphs.js';
const result = executeGlyph('⊕', a, b);
```

### `Tensor`

```js
import { Tensor } from './stdlib/tensor.js';
const t = Tensor.zeros([128, 64]);
```

---

## Tools

### `KuhulLinter`

```js
import { KuhulLinter } from './tools/linter.js';
const { warnings, errors } = new KuhulLinter().lint(source);
```

### `KuhulFormatter`

```js
import { KuhulFormatter } from './tools/formatter.js';
const formatted = new KuhulFormatter().format(source);
```

### `KuhulREPL`

```js
import { KuhulREPL } from './tools/repl.js';
const repl = new KuhulREPL();
await repl.start();              // interactive
const result = await repl.eval('[Pop]\n  [Wo X tensor<float32, [1]>]\n[Xul]');
```
