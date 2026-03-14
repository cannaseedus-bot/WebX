# K'UHUL++ Geometric IR Specification

## Overview

The **Geometric IR** (GIR) is the intermediate representation that sits between the annotated
AST and the code-generation backends. It uses a flat SSA-style instruction list with glyph
operations mapped to geometric manifold M primitives.

## GeometricIR Structure

```typescript
interface GeometricIR {
    instructions: IRInstruction[];
    manifold:     ManifoldDef;
    phases:       Phase[];
    symbols:      Map<string, KuhulType>;
}
```

### ManifoldDef

```typescript
interface ManifoldDef {
    dimensions: number;   // 1–16
    metric:     'euclidean' | 'riemannian' | 'minkowski' | 'π-harmonic';
    phase:      number;   // initial phase in [0, 2π]
}
```

### Phase

```typescript
interface Phase {
    name:         string;
    start:        number;  // radians
    end:          number;  // radians
    instructions: IRInstruction[];
}
```

## Instruction Set

| Op           | Fields                              | Description                     |
|--------------|-------------------------------------|---------------------------------|
| `const`      | id, value, type                     | Load a compile-time constant    |
| `load`       | id, name, type                      | Load a named binding            |
| `store`      | id, name, src                       | Store to a named binding        |
| `alloc`      | id, type                            | Allocate a typed buffer         |
| `invoke`     | id, callee, args[], returnType      | Call a function                 |
| `phase`      | id, delta                           | Advance phase by delta radians  |
| `branch`     | id, target                          | Unconditional jump              |
| `condbranch` | id, cond, ifTrue, ifFalse           | Conditional jump                |
| `label`      | id, name                            | Basic block label               |
| `return`     | id, value?                          | Return from program             |
| `⊗`          | id, left, right, type               | Geometric product               |
| `⊕`          | id, left, right, type               | Translation / bias addition     |
| `⊖`          | id, left, right, type               | Difference                      |
| `⊛`          | id, left, right, type               | Convolution                     |
| `⊜`          | id, left, right, type               | Identity / copy                 |
| `⊝`          | id, left, right, type               | Complement / negation           |
| `⊞`          | id, left, right, type               | Union / element-wise add        |
| `⤍`          | id, left, right, type               | Vector Encrypt (affine)         |
| `↻`          | id, left, right, type               | Rotational Compression          |
| `⟲`          | id, left, right, type               | Spherical Loop                  |
| `∿`          | id, left, right, type               | Torsion Field                   |
| `⊙`          | id, left, right, type               | Radial Projection               |
| `≋`          | id, left, right, type               | Wave Modulation                 |

## Type System

```typescript
type DataType   = 'float32' | 'float64' | 'int32' | 'uint32'
type KuhulType  = TensorType | ScalarType | PhaseType | ManifoldType | StringType

interface TensorType  { kind: 'tensor';   dtype: DataType; shape: number[] }
interface ScalarType  { kind: 'scalar';   dtype: DataType }
interface PhaseType   { kind: 'phase';    value: number }
interface ManifoldType{ kind: 'manifold'; dimensions: number; metric: string }
interface StringType  { kind: 'string' }
```

## Optimization Passes

1. **Constant Folding** — folds `⊕`, `⊖`, `⊗`, `⊞` of numeric constants at compile time
2. **Dead Code Elimination** — removes `const`, `load`, `alloc` whose result is never used
3. **Transform Fusion** — fuses consecutive `⊗` chains into `__fused_matmul`

## Verification Checks

- All operand SSA ids are defined before use
- All branch targets reference defined labels
- All glyph ops use recognised glyph symbols
- Phase ranges are within [0, 2π]
- Manifold dimensions are between 1 and 16

---

*See also: [COMPILER.md](COMPILER.md)*
